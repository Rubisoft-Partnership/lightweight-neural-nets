import serial
import time

# Configuration for the serial port
serial_port = '/dev/ttyUSB0'  # Update this to your correct serial port
baud_rate = 115200            # This should match the baud rate set in your ESP32 code

# Open the serial port
ser = serial.Serial(serial_port, baud_rate, timeout=1)

try:
    while True:
        # Prompt the user to enter a message
        message = input("Enter message to send: ")

        if message.lower() == 'exit':
            print("Exiting...")
            break

        # Send the message to the ESP32
        ser.write(message.encode())

        # Give the ESP32 some time to respond
        time.sleep(0.5)

        # Read the response from the ESP32
        response = ser.read(ser.in_waiting)
        if response:    
            # Decode the response, ignoring errors or replacing them
            decoded_response = response.decode('ISO-8859-1')
            if decoded_response:
                print(f"ESP32 responded:\n{decoded_response}")

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Close the serial port
    ser.close()