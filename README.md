# Open LLMs are Necessary for Private Adaptations and Outperform their Closed Alternatives


**Keywords:** LLM, privacy, differential privacy, dpsgd, LoRA, private fine-tuning, PromptPATE, PromptDPSGD, Adaptations, soft prompt, prefix tuning, hard prompts

**TL;DR:** We demonstrate that privately adapting open LLMs outperforms the recently proposed private adaptations for closed LLMs across all privacy regimes while using smaller model architectures and incurring lower financial costs.

**Abstract:**

While open Large Language Models (LLMs) have made significant progress, they still fall short of matching the performance of their closed, proprietary counterparts, making the latter attractive even for the use on highly \textit{private} data. 
Recently, various new methods have been proposed to adapt closed LLMs to private data without leaking private information to third parties and/or the LLM provider. 
In this work, we analyze the privacy protection and performance of the four most recent methods for private adaptation of closed LLMs. 
By examining their threat models and thoroughly comparing their performance under different privacy levels according to differential privacy (DP), various LLM architectures, and multiple datasets for classification and generation tasks, we find that: (1) all the methods leak query data, i.e., the (potentially sensitive) user data that is queried at inference time, to the LLM provider, (2) three out of four methods also leak large fractions of private training data to the LLM provider while the method that protects private data requires a local open LLM, (3) all the methods exhibit lower performance compared to three private gradient-based adaptation methods for \textit{local open LLMs}, and (4)~the private adaptation methods for closed LLMs incur higher monetary training and query costs than running the alternative methods on local open LLMs.
This yields the conclusion that, to achieve truly \textit{privacy-preserving LLM adaptations} that yield high performance and more privacy at lower costs, one should use open LLMs.

## Code
Our code is divided into mutliple directories, each with their own README. 

The directories inlcude the following:

**PrivateLoRA-Classification** includes our code to run the Private LoRA for the evaluated **classification** tasks \
**PrivateTuning-Generation** includes our code to run the private tuning methods for the evaluated **generation** tasks \
**DP-ICL** includes an adaptation of the code from https://arxiv.org/abs/2305.01639 for **classification** tasks \
**PromptPATEGen** inlcudes our code for PromptPateGen for **generation** tasks
