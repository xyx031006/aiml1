#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("========== EXPERT SYSTEM ==========")

print("\n========== KNOWLEDGE BASE ==========")

rules = [
    (["has_fur", "barks"], "dog"),
    (["has_fur", "meows"], "cat"),
    (["has_feathers", "lays_eggs"], "bird"),
    (["bird", "swims"], "duck"),
    (["bird", "cannot_fly"], "penguin"),
    (["has_scales", "lives_in_water"], "fish"),
    (["dog"], "pet_animal"),
    (["cat"], "pet_animal")
]

facts = set()

def forward_chaining(facts):

    changed = True

    while changed:
        changed = False

        for conditions, conclusion in rules:

            if all(condition in facts for condition in conditions):

                if conclusion not in facts:
                    facts.add(conclusion)
                    print("New Fact Derived:", conclusion)
                    changed = True

    return facts


def backward_chaining(goal, facts, visited=None):

    if visited is None:
        visited = set()

    if goal in facts:
        return True

    if goal in visited:
        return False

    visited.add(goal)

    for conditions, conclusion in rules:

        if conclusion == goal:

            if all(
                backward_chaining(condition, facts, visited.copy())
                for condition in conditions
            ):
                return True

    return False


print("\nEnter Animal Properties")
print("Examples: has_fur, barks, meows, has_feathers")
print("Type 'done' when finished.")

while True:

    fact = input("\nEnter Fact: ").lower().strip()

    if fact == "done":
        break

    if fact:
        facts.add(fact)

print("\n========== FORWARD CHAINING ==========")

print("Initial Facts:", facts)

derived_facts = forward_chaining(facts)

print("\nFinal Knowledge Base:")

for fact in sorted(derived_facts):
    print("-", fact)

print("\n========== BACKWARD CHAINING ==========")

goal = input("\nEnter Goal to Check: ").lower().strip()

if backward_chaining(goal, derived_facts):
    print("\nGoal Proved:", goal)
else:
    print("\nGoal Cannot Be Proved:", goal)

print("\nExpert System Execution Completed!")


# In[ ]:




