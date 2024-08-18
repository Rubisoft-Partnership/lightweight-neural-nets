import serial
import time
from sklearn import datasets

# Configuration for the serial port
SERIAL_PORT = '/dev/ttyUSB0'  # Update this to your correct serial port
BAUD_RATE = 115200            # This should match the baud rate set in your ESP32 code

# Open the serial port
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Load the digits dataset
digits, targets = datasets.load_digits(return_X_y=True)


def read_serial_response() -> str:
    """Read and return the response from the serial port."""
    response = ser.read(ser.in_waiting)
    if response:
        try:
            return response.decode('ISO-8859-1')
        except UnicodeDecodeError:
            return "Invalid response received."
    return ""


def send_feature_data(digit, target):
    """Send digit features and target one-hot encoding over serial."""
    for feature in digit:
        ser.write(f"{feature/16:.4f} ".encode())
        time.sleep(0.2)  # Adjust delay based on ESP32 processing speed

    one_hot = [0] * 10
    one_hot[target] = 1
    for label in one_hot:
        ser.write(f"{label} ".encode())
        time.sleep(0.2)  # Adjust delay based on ESP32 processing speed

    ser.write("DONE.".encode())


def process_data():
    """Main loop to process and send data based on ESP32 readiness."""
    batch_size = 0
    while True:
        time.sleep(1)  # Wait a little before checking again
        response = read_serial_response()
        if "READY" in response:
            batch_size = int(response.split("READY.")[
                             1].split("BS:")[1].split("\n")[0])
            print(f"ESP32 ready to receive {batch_size} samples.")
            break
    sample_num = 1
    for digit, target in zip(digits, targets):
        print(f"Sending sample {sample_num}...")
        send_feature_data(digit, target)
        print(f"Sent sample {sample_num}")
        sample_num += 1

        if sample_num > batch_size:
            print("All samples sent.")
            break
        time.sleep(1)  # Wait a little before checking again


def main():
    try:
        process_data()
        # Then print the response from the ESP32
        while True:
            time.sleep(1)
            response = read_serial_response()
            if response:
                print(response)
    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt.")
    finally:
        ser.close()
        print("Serial port closed.")


if __name__ == "__main__":
    main()
