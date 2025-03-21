import network
import urequests
import time
import dht
from machine import Pin, ADC

# Wi-Fi Credentials
WIFI_SSID = "your SSID"
WIFI_PASSWORD = "your password"

# Firebase Config
FIREBASE_URL = "Enter firebase url"      
FIREBASE_AUTH = "Enter Firebase Auth tocken"

# Sensor Pins
DHT_PIN = 4  # Change based on your setup
MOISTURE_PIN = 35  # Analog pin for moisture sensor

# Initialize Sensors
dht_sensor = dht.DHT11(Pin(DHT_PIN))
moisture_sensor = ADC(Pin(MOISTURE_PIN))
moisture_sensor.atten(ADC.ATTN_11DB)  # Full range (0-3.3V)

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
        response = urequests.put(url, json=data, headers=headers)
        print("Firebase Response:", response.text)
        response.close()
    except Exception as e:
        print("Error sending data:", e)

# Main Loop
connect_wifi()
while True:
    try:
        dht_sensor.measure()
        temp = dht_sensor.temperature()
        hum = dht_sensor.humidity()
        moisture = moisture_sensor.read()  # 0-4095 (ESP32 ADC range)
        
        print(f"Temp: {temp}°C, Humidity: {hum}%, Moisture: {moisture}")
        send_to_firebase(temp, hum, moisture)
        
    except Exception as e:
        print("Error reading sensors:", e)
    
    time.sleep(5)  # Send data every 10 seconds

