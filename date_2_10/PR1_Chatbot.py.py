#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("========== PEAS REPRESENTATION ==========")

print("Performance Measure: Accurate and helpful responses")
print("Environment: User and Chat Interface")
print("Actuators: Displaying chatbot responses")
print("Sensors: Receiving user queries")


def chatbot(user_input):

    message = user_input.lower()

    if "hello" in message or "hi" in message:
        return "Hello! Welcome to Intelligent Chatbot."

    elif "your name" in message:
        return "I am an Intelligent Rule-Based Chatbot."

    elif "how are you" in message:
        return "I am fine. How can I help you?"

    elif "python" in message:
        return "Python is a popular programming language."

    elif "ai" in message:
        return "AI stands for Artificial Intelligence."

    elif "college" in message:
        return "College is a place where students learn new skills."

    elif "thank" in message:
        return "You are welcome!"

    elif "bye" in message:
        return "Goodbye! Have a nice day."

    else:
        return "Sorry, I don't understand. Please ask another question."


print("\n========== CHATBOT STARTED ==========")
print("Type 'bye' to stop the chatbot.")

while True:

    user_input = input("\nYou: ")

    response = chatbot(user_input)

    print("Chatbot:", response)

    if user_input.lower().strip() == "bye":
        break

print("\nChatbot Session Ended Successfully!")


# In[ ]:




