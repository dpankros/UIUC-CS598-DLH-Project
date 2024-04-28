# CS 598 DLH Final Project Video Outline

The following is the prompt for this video:

---
**no need to make slides**. We expect a well-timed, well-presented presentation. You should clearly explain what the original paper is about (what the general problem is, what the specific approach taken was, and what the results claimed were) and what you encountered when you attempted to reproduce the results. You should use the time given to you and not too much (or too little).

* <= 4 mins
* Explain the general problem clearly
* Explain the specific approach taken in the paper clearly
* Explain reproduction attempts clearly

## Outline

### The problem

- Obstructive sleep apnea hypopnea syndrome (OSAHS), commonly called sleep apnea, is linked with bad side effects that can last for life. If you can catch it in kids, you can treat it way more effectively
- Diagnosing sleep apnea is hard - the state of the art right now is to go to a hospital and do an all-night sleep study
- Sleep labs are hard to find (esp in rural areas)
- Even when you are near a sleep lab, it's hard to do a sleep study on kids

### Approach taken by the paper

- Get signals data that's easier/cheaper to get than a sleep study
    - Nationwide Children's Hospital (NCH) Sleep Data Bank
    - Childhood Adenotonsillectomy Trial (CHAT) dataset
- Train a transformer-based deep learning model on those data, and compare to other baseline models (models that have been used for this purpose in previous research)
- Run a series of ablations, including for signals that are easy to get at home -- ECG and SpO2

### Our paper reproduction attempts

- Took necessary trainings and applied for data access, then when granted access, began data download
- Downloaded the code from the paper's open-source repository
- Figured out the required python version and package dependencies in the codebase
- Spent time getting the code running properly, adding documentation and other small necessary utilities
- Added code to generate apnea-hypopnea index (AHI) scores for all sleep studies
- Spent time on cleaning / speeding up training loop
- Finished training and running simple testing logic in the original repo
- Ran repo-included ablations
- Built evaluation code, including visualization/graphing logic
- Did significantly more ablations than what was included
- Did deeper statistical analyses on the results of our extensive ablation suite
- Finally translated our on-filesystem code to the notebook (i.e. linearize dependencies, etc...)
