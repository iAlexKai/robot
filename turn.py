import serial
serial_agent = serial.Serial("/dev/ttyUSB0", 9600)

outputCommand = bytes('$AP0:175X254Y127A127B!', encoding='utf8')

serial_agent.write(outputCommand)
