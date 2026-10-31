# Phase 1 Technical Report — Draft for Student Review

**DSAI 403: Nature-Inspired Computation · Project 08 · Zewail City of Science, Technology and Innovation**

**Instructor:** Assoc. Prof. Mohamed Maher Ata

**Prepared:** 31 October 2026
**Status:** Phase 1 experiment completed; report draft for student review.

> **Source and submission note.** The course file `NIC_Project8_PSO_UNet_Arabic_English.pdf` was not present in the supplied project workspace. This draft follows the attached project brief and the cited primary literature. The student team must compare the document against the official course handout, verify every claim and result, revise the prose in their own voice, and check the course rules for AI-assisted work before submission.

## 1. Project Title

**Particle Swarm Optimization of U-Net Hyperparameters for Pet Image Segmentation.** This Phase 1 project implements a reproducible binary semantic segmentation experiment using a compact U-Net, the Oxford-IIIT Pet dataset, and Particle Swarm Optimization (PSO). It evaluates a fixed-hyperparameter baseline against an 8-epoch U-Net trained from scratch with hyperparameters selected by a short PSO pilot. It is an implementation and evaluation of established techniques on a selected dataset, not a claim of a new optimization algorithm.

## 2. Problem Definition

Given an RGB photograph containing a cat or dog, binary semantic segmentation assigns each pixel either pet foreground or non-pet background. This differs from breed classification: the output is a spatial mask rather than one label for the whole photograph. Segmenting pets is difficult because backgrounds may be cluttered, contours are irregular, fur creates thin structures, lighting and object scale vary, and some foreground colors resemble the background. Under a limited compute budget, network training is also sensitive to learning rate, regularization, and the number of candidate experiments that can be afforded.

The research question is: **Can Particle Swarm Optimization select U-Net hyperparameters that improve pet image segmentation performance relative to a fixed baseline while maintaining a practical computational budget?** Improvement is not presumed. A meaningful comparison requires the same data split, preprocessing, loss, metric policy, random-seed policy, final training epochs, and checkpoint-selection rule. The additional PSO search cost must be reported separately from final model training time.

The primary quality measures are mean foreground Dice coefficient and mean foreground intersection-over-union (IoU). Secondary measures are baseline runtime, PSO search time, and final training time. The final held-out test set should be used once the training protocol and hyperparameters are fixed; all model selection uses validation data only.

## 3. Dataset Description

The [Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) contains photographs across 37 cat and dog breeds with pixel-level trimap annotations [3]. The selected protocol uses 7,349 images: the official 3,680-image development split is divided with seed 42 into 2,944 training and 736 validation members, while all 3,669 official test images are held out. The implementation uses `torchvision.datasets.OxfordIIITPet(..., target_types="segmentation")` and writes exact split memberships to `splits/train.txt`, `splits/validation.txt`, and `splits/test.txt` when data preparation executes. It checks pairing and split non-overlap before training.

Each input image is resized to 128 × 128 pixels with bilinear interpolation and converted to three-channel float values in `[0, 1]`. Each trimap is resized with nearest-neighbor interpolation to preserve integer labels. Label `1` maps to foreground `1`; label `2` maps to background `0`; label `3` is excluded via an explicit valid-pixel mask. A horizontal flip with probability 0.5 is applied jointly to image, binary target, and valid mask only for training. Validation and test transformations are deterministic. The target and valid mask are kept separate so ignored pixels do not influence the loss or reported metrics.

The official dataset page states that the image and annotation archives total roughly 800 MB and that images vary in scale, pose, and lighting. It also specifies the dataset license and retains the original image owners' copyright. The downloaded archives and unpacked data therefore stay outside Git. The Oxford host timed out on the local machine, so the exact archives were obtained from a [mirror](https://huggingface.co/datasets/Shiym/ViT-FineTune/tree/main/oxford-iiit-pet), with MD5 values `5c4f3ee8e5d25df40f4fd59a7f44e54c` (images) and `95a8c909bbe2e81eed6a22bccdf3f68f` (annotations) matching Torchvision's official checksums. Preparation then verified all three expected split counts and zero overlap.

![Figure 5. Original pet images, trimaps, processed binary targets, and valid-pixel masks.](../figures/dataset_samples/examples.png)

## 4. Literature Review

Kennedy and Eberhart [1] introduced particle swarm optimization as a continuous-function optimizer motivated by social behavior. Particle positions evolve using velocity, individual best, and swarm best information. Their work established the search idea, but did not study modern convolutional segmentation or prove that this project's small swarm will improve validation quality. Ronneberger, Fischer, and Brox [2] introduced U-Net's contracting path for context and symmetric expanding path for localization, with skip connections and augmentation. Their original application was biomedical segmentation; this project adapts the architecture at lower width and resolution for pet photographs.

Parkhi and colleagues [3] introduced the pet dataset for fine-grained breed recognition, including trimap segmentation annotations. The approximately 59% breed-classification accuracy reported by the original paper concerns a different task and is not a segmentation baseline. Saifullah and Dreżewski [4] reported PSO tuning of learning rate and dropout for a modified U-Net on brain-tumor MRI; the publisher abstract reports improvements of up to four percentage points in Dice and two in Jaccard over the comparisons it describes. Their 2025 PSO-UNet paper [5] reports tuning filters, kernel size, and learning rate for multimodal MRI segmentation, with abstract-reported Dice values of 0.9578 and 0.9523 on its stated datasets. These domain-specific results motivate the approach but do not predict pet-segmentation performance.

| Source | Research problem and method | Optimized parameters / verified finding | Limitation and relevance here |
|---|---|---|---|
| Kennedy & Eberhart (1995) [1] | Continuous nonlinear optimization with a particle swarm; benchmark tests and prospective neural-network uses. | Established particle-based search. The present inertia and coefficient settings are project pilot choices, not claimed optimal values from [1]. | No U-Net or pet segmentation; supplies the NIC foundation. |
| Ronneberger et al. (2015) [2] | Dense biomedical segmentation with a contracting and symmetric expanding convolutional network and data augmentation. | Source reports strong performance on its biomedical challenge; no PSO tuning. | Different imaging domain and network scale; supplies the encoder-decoder design. |
| Parkhi et al. (2012) [3] | Fine-grained cat/dog breed classification and dataset creation. | 37 breeds; images include trimap masks; reported breed accuracy is about 59%, not segmentation Dice. | Provides the selected dataset, not a directly comparable segmentation result. |
| Saifullah & Dreżewski (2024) [4] | Brain-tumor MRI segmentation with modified U-Net and PSO before training. | Tunes learning rate and dropout; publisher abstract reports up to +4 percentage points Dice and +2 Jaccard in its comparisons. | Medical domain and different protocol; supports the tuning hypothesis only. Full chapter is subscription content. |
| Saifullah & Dreżewski (2025) [5] | Multimodal brain-tumor segmentation with PSO-UNet. | Tunes filter count, kernel size, and learning rate; arXiv abstract reports Dice 0.9578 and 0.9523 on BraTS 2021 and Figshare. | Different search space, architecture, data, and compute; no score transfer to pets. |

The literature provides architecture and search precedents. The present work tests a more constrained search over learning rate, bottleneck dropout, and weight decay. A held-out test comparison checks whether the selected validation advantage persists in this run; repeat seeds are still needed to estimate its reliability.

## 5. Selected NIC Algorithms

PSO is the selected Nature-Inspired Computation method. Each particle represents three continuous training hyperparameters:

```text
x = [log10(learning_rate), dropout, log10(weight_decay)]
learning_rate in [0.0001, 0.003]
dropout in [0.0, 0.3]
weight_decay in [0.000001, 0.001]
```

Logarithmic coordinates give multiplicative parameters more balanced exploration. At each round, a candidate initializes a **fresh** U-Net, trains on the training split for two epochs, and returns mean validation Dice. The minimized objective is `fitness(x) = 1 − validation_Dice(x)`. The official test set never enters this objective.

For particle `i`, dimension `d`, and iteration `t`, the selected update is:

```text
v[i,d,t+1] = w*v[i,d,t]
             + c1*r1*(pbest[i,d] - x[i,d,t])
             + c2*r2*(gbest[d] - x[i,d,t])
x[i,d,t+1] = clip(x[i,d,t] + v[i,d,t+1], lower[d], upper[d])
```

The pilot defaults are `w = 0.7`, `c1 = 1.5`, `c2 = 1.5`, seed 42, three particles, and two evaluation rounds **including initialization**. Thus a complete pilot has six candidate fitness evaluations and a total of twelve candidate-training epochs. This is a limited search, not evidence of a globally optimal configuration. Candidate quality and total search time are logged per run; a fresh final U-Net is then trained for eight epochs to match the baseline final budget. Search time is a separate cost.

## 6. Selected AI/Deep Learning Model

The compact U-Net consumes `(B, 3, 128, 128)` tensors. Four encoder double-convolution blocks use 16, 32, 64, and 128 output channels, with max pooling between levels. A 256-channel double-convolution bottleneck contains the only configurable dropout. The decoder upsamples and concatenates same-scale encoder features through skip connections, then applies double-convolution blocks with 128, 64, 32, and 16 output channels. A final `1 × 1` convolution returns `(B, 1, 128, 128)` **logits**. No pretrained weights are used. Dropout placement stays identical for the baseline and PSO candidates; only its probability changes.

For each image, binary cross-entropy with logits is averaged over valid pixels only. Soft Dice uses only valid-pixel probabilities and labels. The common loss is `0.5 × masked BCE + 0.5 × masked soft Dice`. An image with zero valid pixels is excluded from the batch average. A whole batch with zero valid pixels does not trigger an optimizer update. The same implementation and numerical weights are used in both experiment branches.

At inference, sigmoid converts logits to probabilities, which are thresholded at 0.5. For each image, only valid pixels contribute to true positives (`TP`), false positives (`FP`), and false negatives (`FN`):

```text
Dice = 2*TP / (2*TP + FP + FN)
IoU  = TP / (TP + FP + FN)
```

If both prediction and truth contain no foreground, both scores equal 1. Images with no valid pixels are excluded. Reported metrics are arithmetic means of the remaining image scores, not global pooled-pixel ratios. This policy avoids a large image or background region dominating the evaluation and is identical for all models.

## 7. Proposed Architecture/Pipeline

The system has a shared data path and two training branches. The shared path downloads or reuses the Oxford-IIIT Pet files, verifies image/mask pairing, creates fixed memberships, and returns image, target, and valid-mask tensors. The baseline branch trains one U-Net with the fixed configuration. The PSO branch evaluates six two-epoch candidates, updates particle bests, exports the best parameter vector, and trains a fresh eight-epoch final U-Net. Both branches select the checkpoint with highest validation Dice. Final evaluation loads each selected checkpoint and computes held-out test Dice and IoU without further tuning.

The editable diagrams are [`overall_pipeline.mmd`](../figures/architecture/overall_pipeline.mmd), [`unet_architecture.mmd`](../figures/architecture/unet_architecture.mmd), [`pso_flow.mmd`](../figures/architecture/pso_flow.mmd), and [`preprocessing_pipeline.mmd`](../figures/architecture/preprocessing_pipeline.mmd). They specify the overall experiment, model geometry, optimizer feedback, and trimap preprocessing. They are methodology diagrams; any plotted performance figures must come from real completed training.

## 8. Initial Implementation

The codebase separates configuration, dataset and preprocessing, U-Net, losses, metrics, common training and evaluation, PSO, visualization, and command-line scripts. `config.json` contains the selected Phase 1 defaults. `python -m scripts.prepare_dataset` prepares local data and split files; `python -m scripts.train_baseline` runs the fixed training protocol; `python -m scripts.run_pso` performs the pilot and final retraining; `python -m scripts.evaluate_models` requires both finalized checkpoints before opening the held-out test split; and `python -m scripts.generate_results` prepares measured result views. The Colab notebook calls these shared entry points rather than maintaining a second training implementation.

The CPU test suite uses synthetic images and masks so ordinary CI does not download the roughly 800 MB dataset or run a full experiment. Tests cover trimap conversion, deterministic splits, model shape and gradients, ignored-pixel loss behavior, empty-valid batches, metric corner cases, PSO bounds and best updates, protocol guards, and a small training integration loop. **Forty local pytest tests passed** on Python 3.11.9 with PyTorch 2.5.1+cpu. The Python 3.11 CPU [GitHub Actions run](https://github.com/Hossam-elsabbagh/PSO-UNet-Image-Segmentation/actions/runs/41126969467) for the initial implementation commit completed successfully, including the dependency-install and synthetic-test steps. A synthetic baseline smoke run and synthetic PSO integration run each completed one small epoch, without being counted as Oxford results.

Reproducibility depends on fixed split IDs, a recorded seed, the executed configuration, compute device, completed epoch count, runtime, and saved history. Checkpoints and downloaded data remain local; histories, small figures, and metadata can be committed if real and reasonably sized. The missing course PDF remains a documentation blocker pending student review.

## 9. Baseline Model

The selected baseline is the compact U-Net described above with AdamW, learning rate `0.001`, weight decay `0.0001`, bottleneck dropout `0.10`, batch size `16`, image size `128 × 128`, seed `42`, and eight epochs. The loss weights are `0.5` for masked BCE and `0.5` for masked soft Dice. After each epoch, the model is evaluated on the validation split; the checkpoint with highest mean validation Dice is saved. Training loss, validation loss, validation Dice, validation foreground IoU, actual epochs, and elapsed training time are recorded by the run. The best checkpoint is selected from validation, not test, performance.

| Parameter | Baseline value | PSO final-model rule |
|---|---:|---|
| U-Net widths | 16 / 32 / 64 / 128; bottleneck 256 | Identical architecture |
| Image size / batch size | 128 × 128 / 16 | Identical |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.001 | PSO-selected within bounds |
| Bottleneck dropout | 0.10 | PSO-selected within bounds |
| Weight decay | 0.0001 | PSO-selected within bounds |
| Loss | 0.5 masked BCE + 0.5 masked Dice | Identical |
| Final epochs / seed | 8 / 42 | 8 / 42 |
| Checkpoint selection | Best validation Dice | Best validation Dice |

The PSO pilot itself trains each candidate for two epochs and uses validation Dice only as a search signal. It must not be put into a comparison table against the eight-epoch baseline as if both were final trained models.

## 10. Experimental Results

The following table uses completed run artifacts only. Each validation row summarizes the best checkpoint chosen across eight epochs; each test row evaluates that fixed checkpoint on all 3,669 official held-out images. Training times are wall-clock measurements repeated in the test rows for context; PSO search time is additional to final-model training time. Exploratory subset runs are excluded because they used different sample counts and training budgets.

| Model | Split | Completed epochs | Mean Dice | Mean foreground IoU | Training time | Search time |
|---|---|---:|---:|---:|---:|---:|
| Fixed baseline U-Net | Validation | 8 | 0.8017 | 0.6946 | 1,618.93 s | Not applicable |
| PSO-selected final U-Net | Validation | 8 | 0.8516 | 0.7667 | 1,766.50 s | 2,441.94 s |
| Fixed baseline U-Net | Held-out test | 8 | 0.8084 | 0.7071 | 1,618.93 s | Not applicable |
| PSO-selected final U-Net | Held-out test | 8 | 0.8547 | 0.7733 | 1,766.50 s | 2,441.94 s |

The full baseline completed all eight epochs on the 2,944-image training and 736-image validation splits. Its best checkpoint was selected at epoch 7 with mean validation Dice `0.8017`, mean foreground IoU `0.6946`, and validation loss `0.2788`. Epoch 8's validation Dice declined to `0.7885` while validation loss rose to `0.3001`, supporting the predefined best-validation-checkpoint rule. The saved checkpoint hash and completed-run metadata were verified before the PSO search began. These are validation measurements, not held-out test results.

The configured PSO pilot completed all six two-epoch candidate evaluations in `2,441.94` seconds. Its best short-run validation Dice was `0.7092`, first reached at evaluation 2. The selected vector was learning rate `0.001071759`, bottleneck dropout `0.0282532`, and weight decay `0.000845020`. Evaluation 5 repeated the best vector, and no later candidate improved the best short-run score. These candidate scores are search fitness, not final-model outcomes.

The fresh PSO-selected model completed eight epochs in `1,766.50` seconds and selected epoch 8, with mean validation Dice `0.8516`, mean foreground IoU `0.7667`, and validation loss `0.2160`. Relative to the baseline's best validation checkpoint, these are gains of `0.0498` Dice and `0.0721` IoU. These validation scores were used for model selection and are therefore followed by the independent test comparison.

![Figure 6. Baseline validation images, ground truth, valid pixels, and predictions from the selected checkpoint.](../figures/predictions/baseline_validation.png)

![Figure 7. Six PSO candidate validation scores and the running best.](../figures/training_curves/pso_convergence.png)

The held-out test result supports a gain for the PSO-selected settings in this run: mean Dice increased by `0.0462` (4.62 percentage points) and mean foreground IoU by `0.0662` (6.62 percentage points). Both models were scored on the same 3,669 test images with the same threshold and metric policy. Test Dice was slightly above each model's selected validation Dice, so this result does not show a validation-only gain. The PSO search and final training together took `4,208.44` seconds, `2.60` times the baseline's `1,618.93` seconds; the search itself accounted for `2,441.94` seconds. Thus the measured quality gain incurred a substantial extra compute cost.

![Figure 8. Held-out test Dice and foreground IoU for the two finalized checkpoints.](../figures/training_curves/comparison_test.png)

The example masks show recognizable pet bodies, but some predictions extend into nearby background and miss thin details such as tails or legs. The score averages do not capture every boundary defect. The pilot searched only three particles across two rounds, and evaluation 5 repeated evaluation 2's vector; two-epoch fitness can rank eight-epoch settings imperfectly. A single seed cannot establish a reliable improvement distribution or statistical significance. Future work should repeat the full protocol across seeds, report mean and spread, consider higher image resolution for fine boundaries, and evaluate whether the additional search time is justified by the task's accuracy needs. The official course handout still needs student review against this draft.

![Figure 9. PSO-selected model test examples with original images, ground truth, valid-pixel masks, and predictions.](../figures/predictions/pso_test.png)

**AI-assistance disclosure draft for student review:** “An AI coding assistant helped draft implementation code, tests, diagrams, notebook text, and this report. The student authors reviewed the code, verified cited sources and executed results, corrected errors, and take responsibility for the submitted work.” This text is a suggested disclosure only. The team must adapt it to its actual contribution and the official course academic-integrity policy.

## References

[1] Kennedy, J., & Eberhart, R. (1995). “Particle swarm optimization.” *Proceedings of ICNN'95*, 1942–1948. https://doi.org/10.1109/ICNN.1995.488968. Original paper consulted via an author-hosted [PDF](https://staff.washington.edu/paymana/swarm/kennedy95-ijcnn.pdf).

[2] Ronneberger, O., Fischer, P., & Brox, T. (2015). “U-Net: Convolutional Networks for Biomedical Image Segmentation.” https://arxiv.org/abs/1505.04597.

[3] Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). “Cats and Dogs.” *IEEE Conference on Computer Vision and Pattern Recognition*. https://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/ and https://www.robots.ox.ac.uk/~vgg/data/pets/.

[4] Saifullah, S., & Dreżewski, R. (2024). “Automatic Brain Tumor Segmentation Using Convolutional Neural Networks: U-Net Framework with PSO-Tuned Hyperparameters.” *Parallel Problem Solving from Nature – PPSN XVIII*, 333–351. https://doi.org/10.1007/978-3-031-70071-2_21. Results summarized from the publisher abstract.

[5] Saifullah, S., & Dreżewski, R. (2025). “PSO-UNet: Particle Swarm-Optimized U-Net Framework for Precise Multimodal Brain Tumor Segmentation.” *GECCO '25 Companion*. https://arxiv.org/abs/2503.19152. Results summarized from the arXiv abstract.
