import serial
import ccc
import time

PORT = 'COM3'           # ESP32 연결 포트 (mac이면 '/dev/ttyUSB0' 등으로 변경)
BAUD_RATE = 115200
FILENAME = 'user001_imu.csv'
DURATION = 30           # 수집 시간 (초)

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # ESP32 초기화 대기

with open(FILENAME, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time(ms)', 'aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'])

    print(f"📡 Logging IMU data for {DURATION} seconds...")
    start_time = time.time()

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                data = line.split(',')
                if len(data) == 7:
                    writer.writerow(data)
                    print(data)
            if time.time() - start_time > DURATION:
                print(f"✅ Done. Data saved to '{FILENAME}'")
                break
        except Exception as e:
            print("❌ Error:", e)
            break

ser.close()
