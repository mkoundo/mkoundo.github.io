---
layout: post
title:  Learning Python the Hard Way
categories: [Code, Python]
excerpt: A hard lesson learnt.
---
So here's an issue I stumbled upon today whilst learning the *quicksort algorithm*, Ref [1]. Quicksort is a divide and conquer algorithm. It first divides the input array into two smaller sub-arrays: the low elements and the high elements. It then recursively sorts the sub-arrays. The steps for in-place Quicksort are:

1. Pick an element, called a pivot, from the array.
2. Partitioning: reorder the array so that all elements with values less than the pivot come before the pivot, while all elements with values greater than the pivot come after it (equal values can go either way). After this partitioning, the pivot is in its final position. This is called the partition operation.
3. Recursively apply the above steps to the sub-array of elements with smaller values and separately to the sub-array of elements with greater values.
The base case of the recursion is arrays of size zero or one, which are in order by definition, so they never need to be sorted.

From Ref [1], the average runtime of quicksort with a randomly selected pivot is O(n log n) vs O(n) for a pivot set to the first array item.

The following code produces quicksort runtimes resulting from a randomly chosen pivot and a pivot set to the first element in the unsorted list, for a randomly chosen list of integers.

```python
import random
import time


def qsort(items, pivot_choice):

    if len(items) < 2:
        return items                        # Base case: 0 or 1 items
    elif len(items) == 2:
        return [min(items), max(items)]     # Base case: 2 items

    # Partition the list
    if pivot_choice:
        pivot = random.choice(items)        # choose random pivot
    else:
        pivot = items[0]            # Choose first item in list as pivot

    items.remove(pivot)
    left_array = [item for item in items if item <= pivot]
    right_array = [item for item in items if item > pivot]

    # Recurse partitioned list
    sorted_items = qsort(left_array, pivot_choice)
    sorted_right = qsort(right_array, pivot_choice)
    sorted_items.append(pivot)
    sorted_items.extend(sorted_right)

    return sorted_items


# Create unsorted list from random numbers
# create a set to avoid duplicate entries
unsorted_set = {int(10000*random.random()) for _ in range(1100)}    
unsorted_list = list(unsorted_set)

start_time = time.time()
print(qsort(unsorted_list, False))              # <-------
end_time = time.time()
print(f'First item pivot done in {end_time-start_time}s')

start_time = time.time()
print(qsort(unsorted_list, True))
end_time = time.time()
print(f'Random pivot done in {end_time-start_time}s')
``` 

The problem is this; calling the `qsort` function once produces the expected results. However, calling the function twice, as per the above code snippet, produces an unexpected result: the second `qsort` function returns a sorted list minus one item. Here is a simplified example reproducing this issue:

```python
def remove_elm(items, item):
    items.remove(item)
    return

my_list = [1, 3, 5]

print(my_list)
print(remove_elm(my_list, 3))
print(my_list)
``` 

the above code produces
```
[1, 3, 5]
None
[1, 5]
```
Stepping through the code; first a list with three items is created and printed. This list and a parameter of 3 are passed to the `remove_elm` function as arguments `items` and `item`, respectively. The function removes element `item` from list `items` and returns nothing. Finally, printing out `my_list` shows the list has had an element removed. So why has `my_list` had an element removed? To answer this question, it helps to use the code visualisation [tool](http://www.pythontutor.com/visualize.html#mode=edit) which provides the following:
![](/images/qsort_eg.png)
It becomes apparent that even though `items` is a local variable within the `remove_elm` function, it still points to the very same list in memory as variable `my_list` does. The visualisation tool shows this by drawing arrows from variables `items` and `my_list` to the same list. Hard lesson learnt! Creating a copy of the list keeps the original list intact as shown: 
![](/images/qsort_eg2.png)

References
1. [Grokking Algorithms](https://www.manning.com/books/grokking-algorithms): An illustrated guide for programmers and other curious people, A. Bhargava, 2016.