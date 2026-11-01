#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("========== SMART HOME CONTROLLER ==========")

print("\n========== PEAS REPRESENTATION ==========")

print("Performance Measure: Energy efficiency and user comfort")
print("Environment: Smart Home")
print("Actuators: Lights and Fans")
print("Sensors: Motion Sensor and Temperature Sensor")

def smart_home(motion, temperature):

    if motion == "yes":
        light = "ON"
    else:
        light = "OFF"

    if temperature >= 30:
        fan = "ON"
    else:
        fan = "OFF"

    print("\n========== SMART HOME STATUS ==========")
    print("Motion Sensor:", motion)
    print("Temperature:", temperature, "°C")
    print("Light Status:", light)
    print("Fan Status:", fan)

    if light == "ON" and fan == "ON":
        print("Message: Light and Fan are turned ON.")

    elif light == "ON":
        print("Message: Light is turned ON.")

    elif fan == "ON":
        print("Message: Fan is turned ON.")

    else:
        print("Message: Light and Fan are turned OFF.")

while True:

    motion = input("\nMotion Detected? (yes/no): ").lower()

    if motion == "exit":
        print("Smart Home Controller Stopped.")
        break

    if motion not in ["yes", "no"]:
        print("Please enter yes or no.")
        continue

    try:
        temperature = float(input("Enter Room Temperature (°C): "))
    except ValueError:
        print("Please enter a valid temperature.")
        continue

    smart_home(motion, temperature)


# In[ ]:




