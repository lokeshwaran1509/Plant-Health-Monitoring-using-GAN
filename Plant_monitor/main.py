import network
import urequests as requests
import time
import random
from machine import Pin, SPI
from ST7735 import TFT
import sysfont

# Wi-Fi Credentials
WIFI_SSID = "Enter SSID"
WIFI_PASSWORD = "Enter password"

# LCD pin
tft_CS = 15
tft_RESET = 4
tft_A0 = 26
tft_SDA = 13
tft_SCK = 14

# Firebase Config
FIREBASE_URL = "enter URL"
FIREBASE_AUTH = "Enter AUth token"

# Initialize SPI and TFT display
spi = SPI(1, baudrate=20000000, polarity=0, phase=0, miso=None)
tft = TFT(spi, tft_A0, tft_RESET, tft_CS)
tft.initr()
tft.rgb(True)
tft.fill(TFT.BLACK)

# Connect to Wi-Fi
def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    while not wlan.isconnected():
        print("Connecting to Wi-Fi...")
        time.sleep(1)
    print("Connected to Wi-Fi:", wlan.ifconfig())

# Send data to Firebase
def send_to_firebase(temp, hum, moisture):
    url = f"{FIREBASE_URL}/sensor_data.json?auth={FIREBASE_AUTH}"
    data = {
        "temperature": temp,
        "humidity": hum,
        "moisture": moisture,
        "timestamp": time.time()
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.put(url, json=data, headers=headers)
        print("Firebase Response:", response.text)
        response.close()
    except Exception as e:
        print("Error sending data:", e)

# Main Loop
connect_wifi()
while True:
    try:
        # Generate random values instead of sensor readings
        temp = round(random.uniform(20.0, 40.0), 2)  # Random temp between 20°C - 40°C
        hum = round(random.uniform(30.0, 90.0), 2)   # Random humidity between 30% - 90%
        moisture = random.randint(0, 4095)           # Random moisture (0-4095, ESP32 ADC range)

        print(f"Temp: {temp}°C, Humidity: {hum}%, Moisture: {moisture}")
        send_to_firebase(temp, hum, moisture)

        # Display data on LCD
        tft.fill(TFT.BLACK)  # Clear screen
        tft.text((10, 20), f"Temp: {temp}C", tft.WHITE, sysfont.sysfont)
        tft.text((10, 40), f"Humidity: {hum}%", tft.WHITE, sysfont.sysfont)
        tft.text((10, 60), f"Moisture: {moisture}%", tft.WHITE, sysfont.sysfont)

    except Exception as e:
        print("Error:", e)

    time.sleep(10)  # Send data every 10 seconds
