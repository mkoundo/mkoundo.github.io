15---
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

Here's the quicksort code with the pivot taken as the first element of the unsorted list. 

```python
def qsort(items):

    if len(items) < 2:
        return items                        # Base case: 0 or 1 items
    elif len(items) == 2:
        return [min(items), max(items)]     # Base case: 2 items

    # Partition the list
    pivot = items[0]
    left_array = [item for item in items[1:] if item <= pivot]
    right_array = [item for item in items[1:] if item > pivot]

    # Recurse partitioned list
    sorted_items = qsort(left_array)
    sorted_right = qsort(right_array)
    sorted_items.append(pivot)
    sorted_items.extend(sorted_right)

    return sorted_items


unsorted_list = [3, 5, 2, 1 4]
print(qsort(unsorted_list))
``` 

Using the `random` module, the above code is modified to select a pivot at random. From Ref [1] the average runtime of quicksort with a randomly selected pivot is $$O(n\log n)$$ vs $$O(n)$$ for a pivot set to the first array item.

In order to compare the runtimes resulting from the two pivot choices, the code is modified to:

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
        pivot = random.choice(items)            # choose random pivot for average runtime of O(n log n)
    else:
        pivot = items[0]                        # Choose first item in list as pivot

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
unsorted_set = {int(10000*random.random()) for _ in range(1100)}    # create a set to avoid duplicate entries
unsorted_list = list(unsorted_set)

# removing the pivot in the qsort function also affects
# the unsorted list because unsorted_list and items point to the same
# list in memory. Therefore use a copy:
unsorted_list_copy = unsorted_list.copy()

start_time = time.time()
print(qsort(unsorted_list, False))
end_time = time.time()
print(f'First item pivot done in {end_time-start_time}s')

start_time = time.time()
print(qsort(unsorted_list_copy, True))
end_time = time.time()
print(f'Random pivot done in {end_time-start_time}s')
``` 

References
1. [Grokking Algorithms](https://www.manning.com/books/grokking-algorithms): An illustrated guide for programmers and other curious people, A. Bhargava, 2016.