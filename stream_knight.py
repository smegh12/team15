"""Headless NeuroPawn Knight -> LSL streamer ('NeuroPawnKnight', 8 EXG channels).
Exits with code 1 if no valid samples arrive within ~15 s of streaming."""
import sys
import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamOutlet

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbserial-A5069RR4"
BOARD = BoardIds.NEUROPAWN_KNIGHT_BOARD.value
BoardShim.disable_board_logger()  # hide the per-packet warning spam


def send(board, cmd):
    """Send a config command; the board's reply is binary, so ignore decode errors."""
    try:
        board.config_board(cmd)
    except UnicodeDecodeError:
        pass


params = BrainFlowInputParams()
params.serial_port = PORT
board = BoardShim(BOARD, params)
print(f"Connecting to {PORT} ...")
board.prepare_session()
code = 0
try:
    print("Turning on channels (about 20 s) ...")
    for ch in range(1, 9):
        send(board, f"chon_{ch}_12")
        time.sleep(1)
        send(board, f"rldadd_{ch}")
        time.sleep(1)
        print(f"  channel {ch} on")

    board.start_stream(45000)
    fs = BoardShim.get_sampling_rate(BOARD)
    exg = BoardShim.get_exg_channels(BOARD)[:8]
    info = StreamInfo("NeuroPawnKnight", "EEG", len(exg), fs, "float32", f"knight-{PORT}")
    chans = info.desc().append_child("channels")
    for i in range(len(exg)):
        c = chans.append_child("channel")
        c.append_child_value("label", f"EXG Channel {i}")
        c.append_child_value("unit", "microvolts")
    outlet = StreamOutlet(info)
    print(f"Streaming {len(exg)} channels at {fs} Hz as 'NeuroPawnKnight'. Ctrl+C to stop.")

    sent, t_last, empty = 0, time.time(), 0
    while True:
        time.sleep(0.05)
        data = board.get_board_data()
        if data.shape[1]:
            outlet.push_chunk(data[exg].T.tolist())
            sent += data.shape[1]
        if time.time() - t_last >= 5:
            print(f"  {sent} samples in the last 5 s")
            empty = empty + 1 if sent == 0 else 0
            if empty >= 3:
                print("No valid samples: BrainFlow can't read this board's packets. "
                      "Falling back to knight_lsl_gui.py ...")
                code = 1
                break
            sent, t_last = 0, time.time()
except KeyboardInterrupt:
    pass
finally:
    try:
        board.stop_stream()
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    board.release_session()
    print("Board released.")
sys.exit(code)
