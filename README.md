# Monty Hall Problem
The Monty Hall problem is a brain-teaser, in the form of a probability puzzle, loosely based on the American television game show Let’s Make a Deal and named after its original host, Monty Hall.
## Define a Monty Hall Problem as a Python function
- Input: your initial selection from [0,1,2]; and your decision: {'**stay**', '**switch**'}
- Outcome: `win `or `lose`

### Style
- There are 3 doors, behind which are two goats and a car.
- You pick a door (call it door A). You’re hoping for the car of course.
- Monty Hall, the game show host, examines the other doors (B & C) and opens one with a goat. (If both doors have goats, he picks randomly.)


```python
import numpy as np
import pandas as pd
import random

## Define a Monty Hall Problem as a Python function
def MHGame(door, d):
  ## Step1: Setup the game with the door (with a Car)
  out = np.zeros(3)
  # generate a door position with a car
  door0 = random.randint(0,2)
  out[door0] = 1
  ## Step2: Your selection
  print('Your selection is %d' %door)
  ## Step3: Monty opens a door
  res_door = set([0,1,2]) - set([door0, door])
  open_door = random.choice(list(res_door))
  print('Monty opens the door: %d' %open_door)
  ## Step4: "stay" or "switch"
  if d == 'stay':
    final_door = door
    print('Your decision is to stay...')
    final_out = out[final_door]
  elif d == 'switch':
    print('Your decision is to switch...')
    final_door = list(set([0,1,2]) - set([door, open_door]))[0]
    final_out = out[final_door]
  else:
    print('Please input d with stay or switch.')
  print('The true door is: %d; your final seletion is: %d' %(door0, final_door))
  if final_out == 1:
    print('You win!')
  else:
    print('Sorry, you lose.')
  return final_out
```
### Test the function
`MHGame(door=0, d='switch')`
```
Your selection is 0
Monty opens the door: 2
Your decision is to switch...
The true door is: 1; your final seletion is: 1
You win!
1.0
```

## Statistical simulation
Attempt `MHGame` by 1000 times, with different options, to see the prob of `win`
```
res = {'door': [], 'decision': [], 'out': []}
n_trial = 1000
for i in range(n_trial):
  for d in ['stay', 'switch']:
    for door in [0, 1, 2]:
      out_tmp = MHGame(door=door, d=d)
      res['door'].append(door)
      res['decision'].append(d)
      res['out'].append(out_tmp)
```
```res = pd.DataFrame(res)```
```res.head(5)```

|	door	|decision  |	out |
| ----------- | ----------- | ----------- |
|0|	0|	stay	|1.0
|1|	1	|stay	|0.0
2	|2	|stay|	1.0
3	|0	|switch|	1.0
4	|1	|switch|	0.0
## Analyze the results
- Compute the conditional probability
- Visualize the results
```
for d in ['stay', 'switch']:
  freq_tmp = res[res['decision'] == d]['out'].mean()
  print('The conditional prob given %s is %f' %(d, freq_tmp))
  ```
  ```
  The conditional prob given stay is 0.328000
The conditional prob given switch is 0.676667
```
```
sum_res = res.groupby(['door', 'decision'])['out'].mean()
sum_res
```
|door | decision| |
|-------|--------|-------|
|0     |stay        |0.321
|      |switch     | 0.674
|1     |stay        |0.320
|      |switch      |0.669
|2     |stay        |0.343
|      |switch      |0.687

`Name: out, dtype: float64`
## Histgram: decision matters?
```
import seaborn as sns
sns.displot(data=res, x='out', col='decision', hue='decision', stat='probability')
```
`<seaborn.axisgrid.FacetGrid at 0x7f2fc06f4430>`
![displot](https://i.imgur.com/c0h5Zh7.png)
## Histgram: initial door matters?
```
import seaborn as sns
sns.displot(data=res, x='out', col='door', hue='door', stat='probability')
```
`<seaborn.axisgrid.FacetGrid at 0x7f2fc054f9d0>`
![displot](https://i.imgur.com/8TWZTDf.png)

  ```
  sum_res = res.groupby(['door', 'decision']).mean().reset_index()
sum_res
```
|door |door	|decision	|out|
|------|-------|----------|---------|
|0	|0	|stay	|0.321
1	|0	|switch	|0.674
2	|1	|stay	|0.320
3	|1	|switch	|0.669
4	|2	|stay	|0.343
5	|2	|switch	|0.687
## Probabilistic interpretation[^1]

[^1]: `latex`([tutorial](https://www.youtube.com/watch?v=zqQM66uAig0)) is required, which is optional (but highly recommended) for our course.

A statistical perspective: we want to make a decision $D$: "stay" (=0) or "switch" (=1), to maximize the probability, that is,
$$
\max_{d = 0, 1} \mathbb{P}\big( Y = 1 \big| D = d \big).
$$
Thus, the primary goal is to compute the conditional probability. Clearly, $Y = 1$ is highly depended on your choice in the first stage, let's denote $X \in \{0, 1\}$ as your initial selection with or w/o car. Then, we have
$$
    \mathbb{P}\big( Y = 1 | D =d \big) = \mathbb{P} \big( Y = 1, X = 0 | D =d \big) + \mathbb{P} \big( Y = 1, X = 1 | D =d \big) \\
    = \mathbb{P} \big( Y = 1, | X = 0, D =d \big) \mathbb{P}(X = 0) + \mathbb{P} \big( Y = 1 | X=1, D =d \big) \mathbb{P}(X = 1)
$$
where the first equality follows from the relation between marginal and joint probabilities, and the second equality follows Bayes' theorem. Note that
$$
\mathbb{P}(X = 1) = \mathbb{P}(\text{initially select car}) = 1/3, \ \
\mathbb{P}(X = 0) = \mathbb{P}(\text{initially select goat}) = 2/3.
$$
Now, we compare the difference in the conditional probabilities when making different decisions.

- ==D = 'stay'==

$$
\mathbb{P}( Y = 1 | D = 0 \big) = \frac{2}{3} \mathbb{P} \big( Y = 1 | X = 0, D=0 \big) + \frac{1}{3} \mathbb{P} \big( Y = 1 | X=1, D=0 \big) = \frac{1}{3}.
$$

- ==D = 'switch'==

$$
\mathbb{P}( Y = 1 | D = 0 \big) = \frac{2}{3} \mathbb{P} \big( Y = 1 | X = 0, D=1 \big) + \frac{1}{3} \mathbb{P} \big( Y = 1 | X=1, D=1 \big) = \frac{2}{3}.
$$