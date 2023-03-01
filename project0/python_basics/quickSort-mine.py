def quickSort(arr):
    if len(arr) < 2:
        return arr

    pivot = arr[0]

    before = [x for x in arr[1:] if x <= pivot]
    after = [x for x in arr[1:] if x > pivot]

    return before + [pivot] + after

if __name__ == '__main__':
    arr = [2, 4, 5, 1]
    print(quickSort(arr))
