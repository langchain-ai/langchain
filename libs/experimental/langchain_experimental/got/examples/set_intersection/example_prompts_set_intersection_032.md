# Set Intersection for 32-Element Sets - Prompts and Examples
## Prompt Templates
### GENERATE: split_prompt
Replace `{input}` with the input list/set of numbers to be split.
```
<Instruction> Split the following list of 32 numbers into 2 lists of 16 numbers each, the first list should contain the first 16 numbers and the second list the second 16 numbers.
Only output the 2 lists in the following format without any additional text or thoughts!:
{
    "List 1": [13, 16, 30, 6, 21, 7, 31, ...],
    "List 2": [25, 24, 10, 4, 27, 0, 14, ...]
} </Instruction>

<Example>
Input: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Output:
{
    "List 1": [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22],
    "List 2": [33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
}
</Example>

Input: {input}
```

### GENERATE: intersect_prompt
Replace `{set1}` and `{set2}` with the sets to be intersected.
```
<Instruction> Find the intersection of two sets of numbers. Output only the set of numbers that are present in both sets, no additional text. </Instruction>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Output: [24, 10, 20, 8]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Output: [40, 15, 5, 24, 35, 59, 16, 63]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Output: [115, 35, 90, 117, 63, 40, 30, 33, 15, 89, 50, 12, 2, 84, 7, 57, 96]
</Examples>

Input Set 1: {set1}
Input Set 2: {set2}
```

### AGGREGATE: merge_prompt
Replace `{input1}` and `{input2}` with the lists/sets to be merged.
```
<Instruction> Merge the following 2 lists of length 16 each, into one list of length 32 by appending the second list to the first list.
Only output the final list without any additional text or thoughts! </Instruction>

List 1: {input1}
List 2: {input2}

```

## Complete Example Prompts
### The GoO Summarised

1. Split the second input set into two sub-sets of equal size (split prompt)
2. For each sub-set: Intersect the sub-set with the first input set (intersect prompt) five times; score each sort attempt; keep the best
3. Merge the resulting intersections into one full intersection set (merge prompt) 10 times; score each merge attempt; keep the best

### Input
```
[31, 11, 4, 63, 38, 58, 59, 24, 61, 14, 32, 39, 27, 46, 48, 19, 52, 57, 50, 56, 3, 2, 53, 29, 5, 37, 62, 41, 36, 12, 49, 16]
```

### Step 1 - Prompt
```
<Instruction> Split the following list of 32 numbers into 2 lists of 16 numbers each, the first list should contain the first 16 numbers and the second list the second 16 numbers.
Only output the 2 lists in the following format without any additional text or thoughts!:
{
    "List 1": [13, 16, 30, 6, 21, 7, 31, ...],
    "List 2": [25, 24, 10, 4, 27, 0, 14, ...]
} </Instruction>

<Example>
Input: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Output:
{
    "List 1": [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22],
    "List 2": [33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
}
</Example>

Input: [31, 11, 4, 63, 38, 58, 59, 24, 61, 14, 32, 39, 27, 46, 48, 19, 52, 57, 50, 56, 3, 2, 53, 29, 5, 37, 62, 41, 36, 12, 49, 16]
```

### Step 1 - Response
```
{
    "List 1": [31, 11, 4, 63, 38, 58, 59, 24, 61, 14, 32, 39, 27, 46, 48, 19],
    "List 2": [52, 57, 50, 56, 3, 2, 53, 29, 5, 37, 62, 41, 36, 12, 49, 16]
}
```

### Step 2a - Prompt
```
<Instruction> Find the intersection of two sets of numbers. Output only the set of numbers that are present in both sets, no additional text. </Instruction>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Output: [24, 10, 20, 8]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Output: [40, 15, 5, 24, 35, 59, 16, 63]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Output: [115, 35, 90, 117, 63, 40, 30, 33, 15, 89, 50, 12, 2, 84, 7, 57, 96]
</Examples>

Input Set 1: [11, 60, 1, 49, 21, 33, 14, 56, 54, 15, 23, 40, 45, 22, 7, 28, 20, 46, 51, 6, 34, 37, 3, 50, 17, 8, 25, 0, 35, 47, 18, 19]
Input Set 2: [31, 11, 4, 63, 38, 58, 59, 24, 61, 14, 32, 39, 27, 46, 48, 19]
```

### Step 2a - 5 Responses
```
1. [11, 14, 46, 14, 19]
2. Output: [11, 14, 46, 19]
3. [11, 14, 46, 14, 19]
4. [11, 14, 46, 19]
5. [11, 14, 46, 14, 19]
```

### Step 2b - Prompt
```
<Instruction> Find the intersection of two sets of numbers. Output only the set of numbers that are present in both sets, no additional text. </Instruction>

<Examples>
Input Set 1: [13, 16, 30, 6, 21, 7, 31, 15, 11, 1, 24, 10, 9, 3, 20, 8]
Input Set 2: [25, 24, 10, 4, 27, 0, 14, 12, 8, 2, 29, 20, 17, 19, 26, 23]
Output: [24, 10, 20, 8]

Input Set 1: [26, 40, 42, 57, 15, 31, 5, 32, 11, 4, 24, 28, 51, 54, 12, 22, 33, 35, 7, 13, 2, 59, 8, 23, 43, 16, 29, 55, 25, 63, 21, 18]
Input Set 2: [16, 60, 36, 48, 0, 15, 5, 19, 46, 24, 1, 6, 61, 10, 38, 53, 58, 9, 44, 14, 35, 63, 52, 20, 27, 17, 39, 47, 34, 56, 40, 59]
Output: [40, 15, 5, 24, 35, 59, 16, 63]

Input Set 1: [115, 61, 35, 103, 90, 117, 86, 44, 63, 45, 40, 30, 74, 33, 31, 1, 118, 48, 38, 0, 119, 51, 64, 78, 15, 121, 89, 101, 79, 69, 120, 29, 58, 50, 116, 11, 60, 12, 39, 95, 23, 2, 109, 84, 7, 43, 99, 98, 52, 70, 75, 102, 57, 19, 94, 36, 114, 88, 71, 56, 83, 6, 96, 107]
Input Set 2: [13, 35, 20, 96, 34, 18, 47, 127, 126, 9, 21, 16, 77, 22, 111, 122, 85, 73, 42, 105, 123, 15, 33, 59, 67, 57, 104, 8, 30, 89, 76, 12, 65, 84, 32, 40, 7, 100, 108, 50, 14, 28, 24, 53, 90, 17, 91, 81, 124, 63, 5, 46, 125, 93, 49, 66, 117, 37, 115, 113, 2, 106, 41, 72]
Output: [115, 35, 90, 117, 63, 40, 30, 33, 15, 89, 50, 12, 2, 84, 7, 57, 96]
</Examples>

Input Set 1: [11, 60, 1, 49, 21, 33, 14, 56, 54, 15, 23, 40, 45, 22, 7, 28, 20, 46, 51, 6, 34, 37, 3, 50, 17, 8, 25, 0, 35, 47, 18, 19]
Input Set 2: [52, 57, 50, 56, 3, 2, 53, 29, 5, 37, 62, 41, 36, 12, 49, 16]
```

### Step 2b - 5 Responses
```
1. [50, 56, 49]
2. [50, 56, 37, 49]
3. [56, 49, 37, 3, 50]
4. [50, 56, 37, 49]
5. [50, 56, 37, 49]
```

### Step 3 - Prompt
```
<Instruction> Merge the following 2 lists of length 16 each, into one list of length 32 by appending the second list to the first list.
Only output the final list without any additional text or thoughts! </Instruction>

List 1: [11, 14, 46, 19]
List 2: [56, 49, 37, 3, 50]
```

### Step 3 - 10 Responses
```
1. [11, 14, 46, 19, 56, 49, 37, 3, 50]
2. [11, 14, 46, 19, 56, 49, 37, 3, 50]
3. [11, 14, 46, 19, 56, 49, 37, 3, 50]
4. [11, 14, 46, 19, 56, 49, 37, 3, 50]
5. [11, 14, 46, 19, 56, 49, 37, 3, 50]
6. [11, 14, 46, 19, 56, 49, 37, 3, 50]
7. [11, 14, 46, 19, 56, 49, 37, 3, 50]
8. [11, 14, 46, 19, 56, 49, 37, 3, 50]
9. [11, 14, 46, 19, 56, 49, 37, 3, 50]
10. [11, 14, 46, 19, 56, 49, 37, 3, 50]
```

### Final Result
```
[11, 14, 46, 19, 56, 49, 37, 3, 50]
```
