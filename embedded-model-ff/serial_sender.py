from sklearn import datasets
import serial
import time

# Configuration for the serial port
serial_port = '/dev/ttyUSB0'  # Update this to your correct serial port
baud_rate = 115200            # This should match the baud rate set in your ESP32 code

# Open the serial port
ser = serial.Serial(serial_port, baud_rate, timeout=1)


# Load the digits dataset
(digits, targets) = datasets.load_digits(return_X_y=True)


def get_from_ser() -> str:
    response = ser.read(ser.in_waiting)
    if response:
        # Decode the response, ignoring errors or replacing them
        decoded_response = response.decode('ISO-8859-1')
        if decoded_response:
            return decoded_response
    return "No response"


try:
    while True:
        # Read the response from the ESP32
        print("Reading response...")
        response = ser.read(ser.in_waiting)
        if response:
            # Decode the response, ignoring errors or replacing them
            decoded_response = response.decode('ISO-8859-1')
            if decoded_response:
                print(f"Response: {decoded_response}")
                # If response contains "READY..." then send the data
                if "READY" in decoded_response:
                    # Parse BS: <num> from the response
                    batch_size = int(decoded_response.split("READY.")[
                                     1].split("BS:")[1].split("\n")[0])
                    print(f"Batch size: {batch_size}")
                    sample_num = 1
                    for (digit, target) in zip(digits, targets):
                        print(f"Sending sample {sample_num}")
                        # Convert the target into one-hot encoding
                        for feature in digit:
                            ser.write(str(feature/16).encode())
                            # print(f"Sent: {feature/16}")
                            # print("Recieved: " + get_from_ser())
                            time.sleep(0.1)
                        one_hot = [0] * 10
                        one_hot[target] = 1

                        for label in one_hot:
                            ser.write(str(label).encode())
                            # print(f"Sent: {label}")
                            # print("Recieved: " + get_from_ser())
                            time.sleep(0.1)
                        print(f"Sent sample {sample_num}")
                        sample_num += 1

                        ser.write("DONE.".encode())
                        
                        time.sleep(2)
                        response = get_from_ser()
                        if "Batch full." in response:
                            print("Batch full.")
                            print(f"Got:\n{response}")
                            break
                        # Give the ESP32 some time to process the dat
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Close the serial port
    ser.close()
