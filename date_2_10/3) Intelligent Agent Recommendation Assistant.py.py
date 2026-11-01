#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("========== RECOMMENDATION ASSISTANT ==========")

print("\n========== PEAS REPRESENTATION ==========")

print("Performance Measure: Relevant and useful recommendations")
print("Environment: User and Recommendation System")
print("Actuators: Displaying recommended items")
print("Sensors: Collecting user preferences")

movies = [
    {"name": "Avengers", "category": "action"},
    {"name": "Inception", "category": "sci-fi"},
    {"name": "Titanic", "category": "romance"},
    {"name": "3 Idiots", "category": "comedy"},
    {"name": "Interstellar", "category": "sci-fi"}
]

books = [
    {"name": "Wings of Fire", "category": "inspiration"},
    {"name": "Rich Dad Poor Dad", "category": "finance"},
    {"name": "Harry Potter", "category": "fantasy"},
    {"name": "The Alchemist", "category": "inspiration"}
]

products = [
    {"name": "Laptop", "category": "technology"},
    {"name": "Smartphone", "category": "technology"},
    {"name": "Headphones", "category": "electronics"},
    {"name": "Running Shoes", "category": "sports"}
]

def recommend(items, preference):
    found = False

    for item in items:
        if item["category"] == preference:
            print("-", item["name"])
            found = True

    if not found:
        print("No matching recommendations found.")

while True:

    print("\n========== MAIN MENU ==========")
    print("1. Movies")
    print("2. Books")
    print("3. Products")
    print("4. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        print("\nAvailable Categories: action, sci-fi, romance, comedy")
        preference = input("Enter your movie preference: ").lower()

        print("\nRecommended Movies:")
        recommend(movies, preference)

    elif choice == "2":
        print("\nAvailable Categories: inspiration, finance, fantasy")
        preference = input("Enter your book preference: ").lower()

        print("\nRecommended Books:")
        recommend(books, preference)

    elif choice == "3":
        print("\nAvailable Categories: technology, electronics, sports")
        preference = input("Enter your product preference: ").lower()

        print("\nRecommended Products:")
        recommend(products, preference)

    elif choice == "4":
        print("\nThank you for using Recommendation Assistant!")
        break

    else:
        print("Invalid choice. Please try again.")

print("\nRecommendation Assistant Stopped.")


# In[ ]:




