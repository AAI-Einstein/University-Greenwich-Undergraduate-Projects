import random


class linkedListNode:

    def __init__(self, value, next_node=None):
        self. value = value
        self.nextNode = next_node


class linkedList:

    def __init__(self, head=None):
        self.head:linkedListNode or None = head

    def insert(self, value):
        node = linkedListNode(value)
        if self.head is None:
            self.head = node
            return

        currentNode: linkedListNode = self.head
        while True:
            if currentNode.nextNode is None:
                currentNode.nextNode = node
                break
            currentNode = currentNode.nextNode

    def __str__(self):
        string = ""
        currentNode: linkedListNode = self.head
        while currentNode is not None:
            string += f"{currentNode.value}->"
            currentNode = currentNode.nextNode
        string += "None"
        return string


ll = linkedList()
ll.insert(3)
print(ll)

for i in range(23):
    ll.insert(random.randint(0, 99))

print(ll)
