## MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation

Changli Wu 1 , 2 , † , Haodong Wang 1 , † , Jiayi Ji 1 , Yutian Yao 5 , Chunsai Du 4 , Jihua Kang 4 , Yanwei Fu 3 , 2 , Liujuan Cao 1 , ∗

1 Xiamen University 2 Shanghai Innovation Institute 3 Fudan University 4 ByteDance 5 Tianjin University of Science and Technology

{ wuchangli, wahadon } @stu.xmu.edu.cn, jjyxmu@gmail.com, 22201316@mail.tust.edu.cn, { duchunsai, kangjihua } @bytedance.com, yanweifu@fudan.edu.cn, caoliujuan@xmu.edu.cn

Flow chart

Figure 1. The Reality Gap: From Idealized 3D RES to Real-World MV-3DRES. (a) Traditional 3D RES depends on dense, high-quality point clouds produced by slow offline scanning and heavy reconstruction. (b) Applied to sparse, low-quality point clouds from real-world RGB views, these models fail to generalize. (c) We introduce MV-3DRES, which uses sparse multi-view RGB inputs and text to achieve robust joint reconstruction and perception, enabled by our MVGGT model.

<!-- image -->

## Abstract

Most existing 3D referring expression segmentation (3DRES) methods rely on dense, high-quality point clouds, while real-world agents such as robots and mobile phones operate with only a few sparse RGB views and strict latency constraints. We introduce Multi-view 3D Referring Expres-

† Equal Contribution.

∗ Corresponding Author.

sion Segmentation (MV-3DRES), where the model must recover scene structure and segment the referred object directly from sparse multi-view images. Traditional two-stage pipelines, which first reconstruct a point cloud and then perform segmentation, often yield low-quality geometry, produce coarse or degraded target regions, and run slowly. We propose the Multimodal Visual Geometry Grounded Transformer (MVGGT), an efficient end-to-end framework that integrates language information into sparse-view geometric reasoning through a dual-branch design. Train-

ing in this setting exposes a critical optimization barrier, termed Foreground Gradient Dilution (FGD), where sparse 3D signals lead to weak supervision. To resolve this, we introduce Per-view No-target Suppression Optimization (PVSO), which provides stronger and more balanced gradients across views, enabling stable and efficient learning. To support consistent evaluation, we build MVRefer, a benchmark that defines standardized settings and metrics for MV-3DRES. Experiments show that MVGGT establishes the first strong baseline and achieves both high accuracy and fast inference, outperforming existing alternatives. Code and models are publicly available at https: //sosppxo.github.io/mvggt.github.io/ .

## 1. Introduction

Grounding natural language in 3D physical scenes is fundamental to embodied AI. A prominent formulation is 3D referring expression segmentation (3DRES), where a model segments an object in a 3D scene given a textual description. Although recent methods [1, 3, 17, 46-48, 53, 55] have achieved strong results, they are built upon a rarely questioned assumption: the availability of dense, complete, and reliable point clouds. Such point clouds typically require LIDAR sensors or lengthy RGB-D SLAM pipelines like BundleFusion [5], which demand deliberate scanning and heavy offline processing. This assumption stands in stark contrast to real-world agents-robots, AR glasses, mobile devices-that perceive environments through only a few casually captured RGB views.

In real settings, high-fidelity geometry is the exception. Sparse multi-view images produce 3D reconstructions that are noisy, incomplete, and often ambiguous (Figure 1(b)). Existing 3DRES models, trained on idealized point clouds, collapse under such inputs. This exposes a fundamental limitation: current 3DRES is sensor-privileged and misaligned with the actual sensing conditions of embodied systems. It motivates a central question: How can we achieve language-grounded 3D perception when complete geometry is no longer given but must be inferred from sparse, inconsistent views?

We address this by introducing Multi-view 3D Referring Expression Segmentation (MV-3DRES), a new setting where the model must jointly reconstruct the scene and segment the referred object directly from sparse RGB views (Figure 1(c)). MV-3DRES is inherently challenging: the model must reason over missing structure, integrate information across misaligned viewpoints, and resolve linguistic ambiguities without access to dense 3D input.

Conventional pipelines fail in this regime. Purely 2D methods cannot enforce global 3D consistency, since they operate on isolated views and cannot resolve depth ordering, occlusion relationships, or spatial relations such as 'in

Flow chart

Figure 2. Illustration of Foreground Gradient Dilusion Problem of Global 3D DICE loss and Per-view No-Target Suppression Optimization.

<!-- image -->

front of' or 'on the left.' As a result, back-projecting their per-view masks produces fragmented or conflicting 3D predictions. Two-stage 'reconstruct-then-segment' pipelines face a different failure mode: sparse inputs yield point clouds that are noisy, incomplete, and structurally distorted, making it difficult for 3DRES models to recover full object extents. Moreover, running a full reconstruction before segmentation incurs substantial latency, limiting practical deployment.

To this end, we propose the Multimodal Visual Geometry Grounded Transformer (MVGGT), the first endto-end architecture designed specifically for MV-3DRES. MVGGT adopts a dual-branch paradigm: a frozen geometric branch provides camera poses, depth cues, and a coarse structural scaffold, while a multimodal branch injects linguistic cues into sparse-view visual features through crossview, cross-modal attention. This design embodies a key conceptual shift: language is intertwined with geometric reasoning from the start, enabling it to guide evidence aggregation and scene disambiguation long before a complete 3D representation exists.

However, sparse-view learning introduces a fundamental optimization challenge. Sparse multi-view reconstruction produces point clouds in which the target instance is represented by only a very small number of scattered points, far fewer than in the dense point clouds used by conventional 3DRES methods. Under such extreme foreground sparsity, standard 3D losses such as Dice become ineffective: gradients from the target region are overwhelmed by background points, leading to Foreground Gradient Dilution (FGD) and causing the optimization to stagnate in early training-gradients are too small to escape poor local minima, as illustrated in Figure 2. The problem is exacerbated by view-dependent visibility-some views contain clear target evidence, while others provide almost none-making uniform 3D supervision unstable and noisy. To mitigate FGD, we introduce Per-view No-target Suppression Optimization (PVSO), which shifts supervision back to 2D view space where the target occupies a larger and more reliable area. This per-view formulation amplifies meaningful gradients from informative views and suppresses misleading signals from target-absent views, resulting in significantly more stable and effective training.

Finally, to standardize evaluation, we construct MVRefer, the first benchmark defining settings, metrics, and data protocol for MV-3DRES. Extensive experiments show that MVGGT provides a strong baseline and significantly outperforms existing alternatives. Taken together, our contributions are fourfold:

- We identify and formalize MV-3DRES, a new problem setting that aligns 3D grounding with realistic sensing conditions.
- We propose MVGGT, a novel dual-branch architecture unifying geometric scaffolding with cross-view, language-aware perception.
- Weanalyze and address the Foreground Gradient Dilution challenge via PVSO, offering a principled optimization strategy tailored for sparse 3D supervision.
- We construct the MVRefer benchmark, defining standardized settings and metrics for MV-3DRES and providing the first strong baseline.

## 2. Related Work

## 2.1. Traditional 3D Referring Segmentation

3D grounding [11, 18, 19, 21, 30, 33, 36, 54] aims to locate a specific object in a 3D scene based on a unique natural language description [1, 3],which is part of visionlanguage tasks [10, 12, 13, 24, 49]. Following this, the 3D-RES (3D Referring Segmentation) task aims to segment a specific object within a point cloud based on a textual query. The field has evolved from foundational twostage paradigms [17, 34, 51] (relying on object proposals and language matching) to recent end-to-end architectures [14, 15, 26, 27, 30, 34, 46-48] demonstrating high efficacy through advanced cross-modal fusion. Despite this progress, the sub-field shares a critical limitation: the requirement for high-quality, dense point clouds. This expensive geometric data is inaccessible to many real-world embodied agents that must rely on sparse, online RGB captures.

## 2.2. Multi-View Feed-forward Reconstruction

Reconstructing 3D geometry from multi-view RGB images provides a practical solution to the input ambiguity in sparse-view settings. Early feed-forward approaches such as DUSt3R and MASt3R [22, 42] introduced coupled scene representations but required heavy post-processing or inte- gration with classical SfM/SLAM pipelines [8, 9, 31, 32] for unconstrained reconstruction. Later works improved efficiency and stability by replacing classical optimization with Transformer-based latent state propagation, as demonstrated by Spann3R, CUT3R, and MUSt3R [2, 39, 41]. Streaming models like WinT3R [45] further enabled realtime performance via sliding-window processing and global camera token pooling.

More recent architectures-VGGT and its successors [7, 35, 38, 40, 43]-adopt alternating-attention designs to achieve robust generalized reconstruction, while extensions address semantic and multi-task perception [23, 37, 44, 50]. This progress culminates in universal backbones such as MapAnything [20], which produce fully factored, metricaware scene representations.

## 3. MV-3DRES Task and MVRefer Benchmark

## 3.1. Task Formulation

We formalize the Multi-view 3D Referring Segmentation (MV-3DRES) task to align 3D language grounding with the sensing constraints of real-world agents. Instead of assuming access to pre-constructed dense point clouds, the model operates directly on sparse multi-view RGB images.

Given a set of N RGB views I = { I i } N i =1 and a naturallanguage referring expression T , the goal is to learn a function

$$f \colon ( I , T ) \rightarrow ( S ^ { \prime } , M ) ,$$

where S ′ ∈ R K × 3 denotes the reconstructed 3D point cloud containing K points, and M ∈ { 0 , 1 } K is the corresponding 3D binary mask marking the points that belong to the object referred to by T . The model must infer both geometry and semantics from the same sparse observations, without any ground-truth 3D input at inference time.

This formulation introduces challenges not present in standard 3DRES. Sparse multi-view observations generate incomplete and noisy geometry, forcing the model to couple reconstruction and grounding. Spatial relations described in language, such as 'on the left of the chair,' must be resolved across viewpoints with inconsistent visibility. Moreover, the target object often occupies only a small portion of the available views, yielding severe foreground sparsity and weak supervisory signals, which we later characterize as Foreground Gradient Dilution (FGD).

## 3.2. The MVRefer Benchmark

To support systematic evaluation of MV-3DRES, we construct MVRefer, a benchmark built upon ScanRefer [3] and the underlying ScanNet sequences [4]. MVRefer is designed to emulate how an embodied agent perceives a scene through a limited number of casual views.

## 3.2.1. Benchmark Setting

For each language-object pair in ScanRefer [3], we sample N = 8 RGB frames from the raw ScanNet video stream [4] at uniform temporal intervals to approximate sparse, on-thefly observations. Sparse sampling creates a solvability issue: the target may be absent from all selected frames. To ensure each sample remains resolvable, we perform a visibility validation step. If none of the initial eight images contain the target, we replace one no-target frame with a randomly chosen target-visible frame. This guarantees at least one positive view while naturally preserving a high proportion of no-target views, maintaining the difficulty inherent to sparse-view grounding.

## 3.2.2. Evaluation Metrics and Splits

Evaluating MV-3DRES requires metrics that disentangle grounding quality from reconstruction quality. Since both outputs in ( S ′ , M ) are jointly predicted from sparse inputs, reconstruction errors can obscure the model's true grounding ability.

Traditional 3D Metric. We report global 3D mean IoU,

$$\ m I o { U } _ { g | o b a l } = I o { U } ( M , M ^ { * } ) ,$$

where M ∗ denotes the ground-truth mask projected onto the reconstructed point cloud S ′ . Although standard, mIoUglobal entangles segmentation performance with the fidelity of S ′ , making it insufficient as the primary diagnostic measure.

Multi-view Diagnostic Metrics. To isolate grounding behaviors, we reproject the predicted 3D mask M into each view using the known camera intrinsics and extrinsics. Let P i ( M ) denote the projected 2D mask for view i , and P i ( M ∗ ) its ground-truth counterpart. We compute:

$$\ m I o { U } _ { v i e v } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } I o { U \left ( P _ { i } ( M ) , P _ { i } ( M ^ { * } ) \right ) } \, , \quad ( 3 ) \quad S ^ { \prime } .$$

$$\ m I o { U } _ { \text {pos} } = \frac { 1 } { | \mathcal { V } ^ { + } | } \sum _ { i \in \mathcal { V } ^ { + } } I o { U } ( P _ { i } ( M ) , P _ { i } ( M ^ { * } ) ) \, , \quad ( 4 )$$

$$\ m I o { U } _ { \text {neg} } = \frac { 1 } { | \mathcal { V } ^ { - } | } \sum _ { i \in \mathcal { V } ^ { - } } I o { U } ( P _ { i } ( M ) , P _ { i } ( M ^ { * } ) ) \, , \quad ( 5 ) \quad \begin{array} { c c } \text {geometric} \\ \text {the} \, \uparrow \\ \text {at} \, \uparrow \end{array}$$

where V + and V -denote the sets of target-visible and no-target views, respectively. These metrics shed light on grounding precision (mIoUpos) and suppression ability (mIoUneg), both of which are essential for robust performance under sparse supervision.

Difficulty Splits. To evaluate robustness under varying signal sparsity, we define two difficulty splits based on the target's 2D pixel ratio. A sample is categorized as hard if the target occupies less than 5% of pixels in all its visible views, and easy if at least one view contains at least 5% target pixels. This separation allows us to isolate performance differences arising from the strength of view-specific supervision.

## 4. Method

We introduce the Multimodal Visual Geometry Grounded Transformer (MVGGT) , an end-to-end dual-branch framework tailored for the MV-3DRES task, as shown in Figure 3.

## 4.1. The proposed MVGGT

MVGGT is designed to jointly recover 3D geometry and perform language-conditioned segmentation from sparse views. The separation into two branches allows the model to exploit a stable geometric scaffold while learning multimodal representations aligned with the text query.

Inputs and Encoders. Given N input images I = { I i } N i =1 and a referring expression T , each image is encoded by a frame-wise ViT, yielding patch embeddings F vis i ∈ R P × D , where P is the number of patches and D the feature dimension. The text is tokenized and processed by a language encoder to produce word embeddings F lang ∈ R W × D , where W denotes token count.

Frozen Reconstruction Branch. The reconstruction branch is a geometry-aware transformer with L blocks. Each block alternates between frame-level self-attention and global cross-view attention, progressively building view-consistent structural cues. Let F geo ℓ denote the features at block ℓ ∈ { 1 , . . . , L } . These features are fed to a reconstruction decoder that predicts camera poses and depth maps, which are back-projected into a coarse point cloud S ′ . All parameters in this branch remain frozen, ensuring a stable geometric prior across training and removing the need to re-learn 3D geometry from sparse images.

Trainable Multimodal Branch. The multimodal branch contains L multi = L/ 3 transformer blocks. Its goal is to fuse geometric cues with text-conditioned visual features. Since the two branches have different depths, we align their interactions by injecting geometric features from the final L/ 3 blocks of the reconstruction branch into all L multi blocks of the multimodal branch.

Geometric Injection. The multimodal branch contains L multi = L/ 3 blocks, and each of them receives geometric guidance from the fi nal L multi blocks of the reconstruction

Flow chart

Figure 3. Architecture of MVGGT, which comprises a frozen Reconstruction Branch that establishes geometric structure and a trainable Multimodal Branch that integrates language into sparse-view visual reasoning.

<!-- image -->

branch. Concretely, for the l ′ -th multimodal block ( l ′ = 1 , . . . , L multi), we take the geometric feature F geo l from the l -th reconstruction block, where l runs over the last L multi layers in order (i.e., the reconstruction branch's ( L -L multi + 1) -th layer provides geometry to the first multimodal block, the next layer to the second, and so on).

These geometric features are passed through a zeroinitialized 1 × 1 convolution Z [52], which projects them into the multimodal feature space. The input to the l ′ -th multimodal block is then

$$F _ { l ^ { \prime } } ^ { \text {in} } = F _ { l ^ { \prime } - 1 } ^ { \text {out} } + \mathcal { Z } ( F _ { l } ^ { \mathbb { G } \circ } ) ,$$

where F out l ′ -1 is the output of the preceding visual attention and cross attention. It allows the multimodal branch to progressively incorporate higher-level geometric cues without perturbing the pretrained reconstruction backbone. F in l ′ is then fed into the visual attention module to get F vis l ′ .

Language Injection. Within each multimodal block, language injection is implemented through a standard crossattention layer. Let F in l ′ be the visual tokens entering block l ′ . Query, key, and value projections are defined as

$$Q = F _ { l ^ { \prime } } ^ { \text {vis} } W _ { Q } , \quad K = F ^ { \text {lang} } W _ { K } , \quad V = F ^ { \text {lang} } W _ { V } , \quad ( 7 )$$

with W Q , W K , and W V ∈ R D × D learnable matrices. Attention is computed as

$$F_{l^{\prime}}^{\text{out}} = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{D}}\right)V$$

and added back through a residual path. This multi-level injection allows text cues to guide feature aggregation across views.

Decoders and Outputs. Finally, the multimodal features F multi L multi are decoded into per-view masks { M i } N i =1 . Using the depths and camera parameters from the frozen reconstruction branch, these 2D masks are back-projected and aggregated on the reconstructed point cloud S ′ to obtain the final 3D mask M . The resulting multi-view predictions serve as the supervision targets in PVSO (Section 4.3).

## 4.2. Foreground Gradient Dilution

Training under the MV-3DRES setting is fundamentally hindered by the extreme sparsity of foreground points in the reconstructed 3D space. Let the Dice loss for 3D segmentation be

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2I}{U}, \quad I = \sum_{j} p_j g_j, \quad U = \sum_{j} p_j + \sum_{j} g_j$$

where p j is the predicted probability at point j and g j ∈ { 0 , 1 } denotes ground-truth labels. The gradient with respect to p j is

$$\frac { \partial \mathcal { L } _ { \text {Dice} } } { \partial p _ { j } } = \frac { 2 ( I - g _ { j } U ) } { U ^ { 2 } } .$$

During early training, predictions remain small and diffuse, yielding I ≈ 0 . In MV-3DRES, reconstructed point clouds from sparse views are large (often 10 6 -10 7 points) while the target instance typically occupies less than 2% of them. Consequently, the union term U is dominated by background, inflating by several orders of magnitude. For a foreground point ( g j = 1 ),

$$\frac { \partial \mathcal { L } _ { D i c e } } { \partial p _ { j } } \Big | _ { g _ { j } = 1 } \approx - \frac { 2 } { U } ,$$

whose magnitude becomes extremely small when U is large. Empirically, gradients fall to 10 -9 -10 -11 , far below the scale needed to drive meaningful updates. Although foreground points are present, their contribution to optimization becomes negligible, leading to stalled convergence. We refer to this failure mode as Foreground Gradient Dilution (FGD) .

## 4.3. Per-view No-Target Suppression Optimization

To mitigate FGD, we propose Per-view No-Target Suppression Optimization (PVSO) , a view-wise supervision strategy that shifts early learning signals from sparse 3D space to the denser 2D image domain. This modification significantly reduces the imbalance between foreground and background, thereby amplifying effective gradients.

Positive-aware Sampling. Given the view set V of a scene, let V t and V n denote target-visible and no-target views, respectively. PVSO samples a subset V ′ while enforcing a minimum foreground-view ratio

$$\rho _ { t } = \frac { | \mathcal { V } _ { t } | } { | \mathcal { V } ^ { \prime } | }$$

ensuring that each batch contains sufficient positive evidence. This prevents the optimization from collapsing to trivial background predictions.

2D Gradient Concentration. For each sampled view, the predicted 3D mask is projected onto the image plane, and a 2D Dice loss is applied. Since foreground regions typically occupy 10 -15% of pixels in visible views-far larger than their &lt; 2% proportion in 3D point clouds-the 2D Dice denominator U 2D becomes substantially smaller:

$$U _ { 2 D } \ll U _ { 3 D } .$$

Given that the Dice gradient scales as O (1 /U ) , the per-view supervision yields foreground gradients that are 1-3 orders of magnitude larger than in 3D. This concentrated 2D supervision strengthens early training signals.

Suppression of No-Target Views. No-target views often far outnumber target-visible ones. To avoid overwhelming the loss with trivial negatives, PVSO normalizes their contribution using

$$w _ { s } = \frac { 1 } { | \mathcal { V } _ { n } | } .$$

The complete PVSO objective is

$$
\begin{aligned}
L_{\text{PVSO}} = \frac{1}{|\mathcal{V}_t| + 1} \Bigg( &\sum_{i \in \mathcal{V}_t} L_{\text{Dice}}(m_i, M_i^{\text{gt}}) + w_s \sum_{j \in \mathcal{V}_n} L_{\text{Dice}}(m_j, \mathbf{0}) \Bigg),
\end{aligned}
$$

where m i is the predicted 2D mask for view i and M gt i is the corresponding ground-truth mask (empty for no-target views).

Joint Objective. PVSO complements 3D supervision by providing dense, stable gradients during early training. The complete objective is

$$L _ { t o t a l } = L _ { B C E } + \lambda _ { p } L _ { P V S O } ,$$

where λ p balances 2D and 3D signals. This formulation alleviates foreground gradient dilution and yields robust multimodal 3D grounding from sparse views.

## 5. Experiments

## 5.1. Implementation Details and Setup

MVGGT Configuration. We adopt the Pi3 [43] reconstruction backbone (36 blocks), which remains frozen throughout training. We use a frozen Roberta model [28] as the language encoder. The multimodal branch contains L multi = 12 blocks and is optimized end-to-end. Training uses AdamW [29] with a 1 × 10 -4 learning rate, batch size of 16, and 30 epochs on a single NVIDIA 4090 GPU. The PVSO weight λ p is fixed to 1 .

Dataset and Metrics. All evaluations are conducted on the proposed MVRefer benchmark. We follow standard ScanRefer train/validation splits [3]. Alongside mIoUglobal, we report all diagnostic view-level metrics from Section 3.2.2, with special focus on the Hard and Easy subsets, which most directly reflect the foreground sparsity challenge underlying FGD.

## 5.2. Main Results

We compare MVGGT with two representative MV-3DRES baselines: (1) 2D-Lift , which lifts ReferDINO [25] masks into 3D via Pi3 [43]; (2) two-stage , which runs Pi3 [43] reconstruction followed by LESS [27] for referring segmentation.

Table 1 shows that MVGGT consistently outperforms all baselines across difficulty levels. On the Hard split, it achieves 24.4 global mIoU-gains of 16.3 and 18.0 over two-stage and 2D-Lift-and improves view mIoU by 52.3 over two-stage. The Easy split follows the same pattern, reaching 50.1 global mIoU and 70.6 view mIoU. Over the full benchmark, MVGGT attains 39.9 global mIoU and 69.3 view mIoU, exceeding the strongest baseline by 22.1 and 48.9. These gains across all subsets confirm MVGGT's robustness to sparse evidence and ambiguity, and highlight PVSO's role in overcoming FGD.

Table 2 compares MVGGT with traditional 3D-RES and recent MV-3DRES baselines under the standard ScanRefer

Table 1. Performance comparison on the MVRefer benchmark.Metrics are all mIoU under different categories.

| Method    | Hard ( ∼ 40%)   | Hard ( ∼ 40%)   | Hard ( ∼ 40%)   | Hard ( ∼ 40%)   | Easy ( ∼ 60%)   | Easy ( ∼ 60%)   | Easy ( ∼ 60%)   | Unique ( ∼ 19%)   | Unique ( ∼ 19%)   | Unique ( ∼ 19%)   | Unique ( ∼ 19%)   | Unique ( ∼ 19%)   | Multiple ( ∼ 81%)   | Multiple ( ∼ 81%)   | Multiple ( ∼ 81%)   | Multiple ( ∼ 81%)   | Overall   | Overall   | Overall   | Overall   |
|-----------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------------|-------------------|-------------------|-------------------|-------------------|---------------------|---------------------|---------------------|---------------------|-----------|-----------|-----------|-----------|
| Method    | global          | view            | pos             | neg             | global          | view            | pos             | neg               | global            | view              | pos               | neg               | global              | view                | pos                 | neg                 | global    | view      | pos       | neg       |
| two-stage | 8.1             | 8.6             | 22.6            | 10.0            | 25.8            | 28.2            | 45.6            | 21.6              | 27.5              | 41.1              | 52.1              | 30.1              | 16.4                | 15.4                | 32.1                | 13.8                | 18.5      | 20.3      | 35.9      | 16.9      |
| 2D-Lift   | 6.4             | 15.0            | 25.3            | 11.9            | 25.4            | 24.1            | 45.1            | 12.3              | 31.5              | 30.9              | 47.2              | 19.2              | 14.5                | 17.9                | 34.8                | 10.4                | 17.8      | 20.4      | 37.2      | 12.1      |
| Ours      | 24.4            | 67.3            | 31.6            | 78.0            | 50.1            | 70.6            | 52.2            | 81.2              | 65.2              | 82.6              | 64.5              | 90.4              | 33.8                | 66.1                | 39.0                | 77.4                | 39.9      | 69.3      | 44.0      | 79.9      |

Table 2. The results of traditional 3D-RES and MV-3DRES tasks under original ScanRefer setting.

| Method                                              | Unique ( ∼ 19%) Acc@25 Acc@50 mIoU                  | Unique ( ∼ 19%) Acc@25 Acc@50 mIoU                  | Unique ( ∼ 19%) Acc@25 Acc@50 mIoU                  | Multiple ( ∼ 81%) Acc@25 Acc@50 mIoU                | Multiple ( ∼ 81%) Acc@25 Acc@50 mIoU                | Multiple ( ∼ 81%) Acc@25 Acc@50 mIoU                | Overall Acc@25 Acc@50 mIoU                          | Overall Acc@25 Acc@50 mIoU                          | Overall Acc@25 Acc@50 mIoU                          |
|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) | Traditional 3D-RES (with ground-truth point clouds) |
| TGNN [17]                                           | 69.3                                                | 57.8                                                | 50.7                                                | 31.2                                                | 26.6                                                | 23.6                                                | 38.6                                                | 32.7                                                | 28.8                                                |
| 3D-STMN [48]                                        | 89.3                                                | 84.0                                                | 74.5                                                | 46.2                                                | 29.2                                                | 31.1                                                | 54.6                                                | 39.8                                                | 39.5                                                |
| SegPoint [15]                                       | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | 41.7                                                |
| Reason3D [16]                                       | 88.4                                                | 84.2                                                | 74.6                                                | 50.5                                                | 31.7                                                | 34.1                                                | 57.9                                                | 41.9                                                | 42.0                                                |
| RG-SAN [46]                                         | 89.2                                                | 84.3                                                | 74.5                                                | 55.0                                                | 35.4                                                | 37.4                                                | 61.7                                                | 44.9                                                | 44.6                                                |
| LESS [27]                                           | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | 53.2                                                | 29.9                                                | 33.7                                                |
| 3D-LLaVA [6]                                        | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | -                                                   | 43.3                                                |
| MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   | MV-3DRES (sparse RGB inputs only)                   |
| two-stage                                           | 43.8                                                | 20.9                                                | 27.5                                                | 25.1                                                | 12.2                                                | 16.4                                                | 28.7                                                | 13.9                                                | 18.5                                                |
| 2D-Lift                                             | 51.6                                                | 26.0                                                | 31.5                                                | 21.5                                                | 5.6                                                 | 14.5                                                | 27.3                                                | 9.6                                                 | 17.8                                                |
| Ours                                                | 83.6                                                | 74.5                                                | 65.2                                                | 49.2                                                | 33.5                                                | 33.8                                                | 55.9                                                | 41.5                                                | 39.9                                                |

protocol. Despite using only sparse RGB inputs, MVGGT achieves 83.6 Acc@25, 74.5 Acc@50, and 65.2 mIoU on the Unique split-substantially narrowing the gap with full 3D-resourced methods. The improvement is even more pronounced in the Multiple split, where MVGGT reaches 33.8 mIoU, exceeding the two-stage and 2D-Lift baselines by 17.4 and 19.3. These results underscore the capability of MVGGT to perform reliable 3D grounding without access to ground-truth point clouds.

## 5.3. Ablation Studies

We conduct comprehensive ablation studies to validate the effectiveness of each proposed component. All experiments are performed on the MVRefer benchmark.

Impact of Core Components. Table 3 contrasts the complete model with partial variants. When neither MVGGT nor PVSO is used, performance drops sharply, reflecting the difficulty of grounding language with sparse and noisy geometry alone. Introducing PVSO optimization yields a clear improvement, raising overall mIoU global from 26 . 9 to 32 . 0 and mIoU view from 41 . 1 to 47 . 5 , indicating that rebalancing per-view gradients effectively addresses the Foreground Gradient Dilution problem and materially enhances the stability of sparse-view learning. MVGGT also increases stability, particularly on hard scenes (from 12 . 9 to 19 . 0 mIoU global), showing that integrating language into geometric reasoning efficiently guides sparse-view aggregation. The full model combines both advantages, achieving

39 . 9 overall mIoU global and 69 . 3 mIoU view . This synergy reinforces our premise that effective MV-3DRES relies on both geometry-aware multimodal design and an optimization strategy resilient to sparse, uneven supervision.

Table 3. Ablation studies on the core components.

| PVSO    | MVGGT   | Easy ↑      | Easy ↑    | Hard ↑      | Hard ↑    | Overall ↑   | Overall ↑   |
|---------|---------|-------------|-----------|-------------|-----------|-------------|-------------|
| PVSO    | MVGGT   | mIoU global | mIoU view | mIoU global | mIoU view | mIoU global | mIoU view   |
| 2D-Lift | 2D-Lift | 25.4        | 24.1      | 6.4         | 15.0      | 17.8        | 20.4        |
| ×       | ×       | 36.3        | 43.0      | 12.9        | 38.5      | 26.9        | 41.1        |
| ✓       | ×       | 40.7        | 48.9      | 19.0        | 45.4      | 32.0        | 47.5        |
| ✓       | ✓       | 50.1        | 70.6      | 24.4        | 67.3      | 39.9        | 69.3        |

PVSO Component Analysis. Table 4 further analyzes PVSO. Without no-target suppression, random view sampling produces unstable performance, especially on hard scenes. Enabling suppression consistently improves results (from 32 . 4 to 36 . 7 overall mIoU global), confirming that reducing misleading gradients from target-absent views is essential. Hybrid sampling strengthens this effect by ensuring sufficient positive evidence in each batch. A no-target ratio of 0 . 5 achieves the most balanced outcome, reaching 39 . 9 mIoU global and 69 . 3 mIoU view . Ratios that are too low underexploit discriminative negative pairs, while overly high ratios drown out positives, leading to degraded training dynamics. These patterns illustrate the core intuition behind PVSO: stable sparse-view learning emerges when each view contributes information in proportion to its actual visibility of the target.

Table 4. Ablation on PVSO components.

| Sampling   | No-Target Ratio   | No-Target   | Easy ↑      | Easy ↑    | Hard ↑      | Hard ↑    | Overall ↑   | Overall ↑   |
|------------|-------------------|-------------|-------------|-----------|-------------|-----------|-------------|-------------|
| Strategy   |                   | Suppression | mIoU global | mIoU view | mIoU global | mIoU view | mIoU global | mIoU view   |
| Random     | -                 | ×           | 40.8        | 65.1      | 19.7        | 66.51     | 32.4        | 65.6        |
| Random     | -                 | ✓           | 46.2        | 55.8      | 22.4        | 49.3      | 36.7        | 53.2        |
| Hybrid     | 0                 | ✓           | 42.4        | 24.3      | 11.2        | 10.3      | 29.9        | 18.7        |
| Hybrid     | 0.25              | ✓           | 46.5        | 65.2      | 19.4        | 59.6      | 35.7        | 63.0        |
| Hybrid     | 0.5               | ✓           | 50.1        | 70.6      | 24.4        | 67.3      | 39.9        | 69.3        |
| Hybrid     | 0.75              | ✓           | 30.3        | 64.0      | 18.7        | 72.0      | 25.7        | 67.2        |

MVGGT Fusion Architecture. Table 5 evaluates where multimodal fusion is best positioned within the encoder. Early fusion performs the weakest, suggesting that injecting language before geometric evidence has formed can disrupt structural reasoning. Middle fusion offers a modest improvement, but late fusion yields the strongest results

Table

Figure 4. Qualitative comparison on the MVRefer benchmark.

<!-- image -->

( 39 . 9 overall mIoU global ; 69 . 3 mIoU view ). This trend suggests that spatial perception should be established first, with language guiding later refinement rather than early feature alignment. Such late fusion yields more stable and discriminative cross-view representations.

Table 5. Ablation on MVGGT fusion stage.

| Fusion Stage   | Easy ↑      | Easy ↑    | Hard ↑      | Hard ↑    | Overall ↑   | Overall ↑   |
|----------------|-------------|-----------|-------------|-----------|-------------|-------------|
| Fusion Stage   | mIoU global | mIoU view | mIoU global | mIoU view | mIoU global | mIoU view   |
| Early          | 45.7        | 65.9      | 21.6        | 63.4      | 36.1        | 64.9        |
| Middle         | 47.5        | 65.7      | 22.5        | 62.1      | 37.5        | 64.3        |
| Late           | 50.1        | 70.6      | 24.4        | 67.3      | 39.9        | 69.3        |

Multimodal Branch Depth. Table 6 studies the depth of the multimodal branch. A shallow configuration (6 layers) struggles on complex scenes, indicating insufficient capacity for cross-view alignment. Increasing to 12 layers delivers consistent improvements across easy and hard subsets, achieving the best overall accuracy. Expanding further to 16 layers causes a sharp decline, pointing to overfitting and unstable attention patterns under sparse supervision. The results reveal a practical design insight: moderate multimodal depth offers the right balance between expressiveness and regularity, enabling language cues to guide view aggregation without overwhelming the geometric signal.

Table 6. Ablation on layer number of multimodal branch.

| Layers   | Easy ↑      | Easy ↑    | Hard ↑      | Hard ↑    | Overall ↑   | Overall ↑   |
|----------|-------------|-----------|-------------|-----------|-------------|-------------|
| Layers   | mIoU global | mIoU view | mIoU global | mIoU view | mIoU global | mIoU view   |
| 6        | 47.1        | 66.7      | 22.5        | 64.1      | 37.3        | 65.7        |
| 12       | 50.1        | 70.6      | 24.4        | 67.3      | 39.9        | 69.3        |
| 16       | 35.6        | 26.2      | 10.1        | 14.2      | 25.4        | 21.4        |

## 5.4. Qualitative Analysis

Figure 3 highlights MVGGT's ability to maintain coherent 3D grounding under sparse and noisy views. While 2D lifting baselines frequently drift to nearby structures or collapse under occlusion and depth ambiguity, MVGGT produces stable segmentations that follow the intended targets across diverse conditions.

In example (a), the baseline confuses a thin whiteboard with adjacent planar surfaces, whereas MVGGT localizes the correct wall-aligned region by combining geometric cues with linguistic context. In example (b), MVGGT isolates the document organizer within a cluttered shelf, supported by PVSO's balanced per-view supervision. Example (c) contains severe depth noise; MVGGT still recovers the toilet bowl, enabled by late-stage multimodal fusion that refines geometry with textual cues. In example (d), the method distinguishes a narrow curtain from a visually similar wall segment. Finally, in example (e), MVGGT identifies the coffee table despite heavy clutter and partial visibility, demonstrating robust cross-view consistency.

## 6. Conclusion

We introduced MV-3DRES, a new setting that aligns 3D language grounding with the sparse and view-limited conditions encountered by real-world agents. Conventional twostage pipelines degrade severely under such sparsity, motivating MVGGT, a dual-branch architecture that integrates linguistic cues directly into sparse-view geometric reasoning. We further identified the Foreground Gradient Dilution problem and addressed it with Per-view No-target Suppression Optimization strategy, enabling stable and efficient training under extreme sparsity. Together with the MVRefer benchmark and the MVGGT model, this work establishes a unified framework for multimodal 3D grounding and paves a practical path toward more capable embodied perception systems.

## References

- [1] Panos Achlioptas, Ahmed Abdelreheem, Fei Xia, Mohamed Elhoseiny, and Leonidas Guibas. Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes. In ECCV , 2020. 2, 3
- [2] Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jerome Revaud, and Vincent Leroy. Must3r: Multi-view network for stereo 3d reconstruction. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 1050-1060, 2025. 3
- [3] Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In ECCV , 2020. 2, 3, 4, 6
- [4] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR , 2017. 3, 4
- [5] Angela Dai, Matthias Nießner, Michael Zollh¨ ofer, Shahram Izadi, and Christian Theobalt. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration. ACM Transactions on Graphics (ToG) , 36(4): 1, 2017. 2
- [6] Jiajun Deng, Tianyu He, Li Jiang, Tianyu Wang, Feras Dayoub, and Ian Reid. 3d-llava: Towards generalist 3d lmms with omni superpoint transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 3772-3782, 2025. 7
- [7] Kai Deng, Zexin Ti, Jiawei Xu, Jian Yang, and Jin Xie. Vggt-long: Chunk it, loop it, align it-pushing vggt's limits on kilometer-scale long rgb sequences. arXiv preprint arXiv:2507.16443 , 2025. 3
- [8] Bardienus Pieter Duisterhof, Lojze Zust, Philippe Weinzaepfel, Vincent Leroy, Yohann Cabon, and Jerome Revaud. Mast3r-sfm: a fully-integrated solution for unconstrained structure-from-motion. In 2025 International Conference on 3D Vision (3DV) , pages 1-10. IEEE, 2025. 3
- [9] Sven Elflein, Qunjie Zhou, and Laura Leal-Taix´ e. Light3rsfm: Towards feed-forward structure-from-motion. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 16774-16784, 2025. 3
- [10] Hao Fei, Shengqiong Wu, Hanwang Zhang, Tat-Seng Chua, and Shuicheng Yan. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. Advances in neural information processing systems , 37:5720757239, 2024. 3
- [11] Mingtao Feng, Zhen Li, Qi Li, Liang Zhang, XiangDong Zhang, Guangming Zhu, Hui Zhang, Yaonan Wang, and Ajmal Mian. Free-form description guided 3d visual graph network for object grounding in point cloud. In ICCV , 2021. 3
- [12] Xuri Ge, Fuhai Chen, Joemon M Jose, Zhilong Ji, Zhongqin Wu, and Xiao Liu. Structured multi-modal feature embedding and alignment for image-sentence retrieval. In Proceedings of the 29th ACM international conference on multimedia , pages 5185-5193, 2021. 3
- [13] Yunpeng Gong, Liqing Huang, and Lifei Chen. Person reidentification method based on color attack and joint defence. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 4313-4322, 2022. 3
- [14] Shuting He and Henghui Ding. Refmask3d: Languageguided transformer for 3d referring segmentation. In ACM MM , 2024. 3
- [15] Shuting He, Henghui Ding, Xudong Jiang, and Bihan Wen. Segpoint: Segment any point cloud via large language model. In ECCV , 2024. 3, 7
- [16] Kuan-Chih Huang, Xiangtai Li, Lu Qi, Shuicheng Yan, and Ming-Hsuan Yang. Reason3d: Searching and reasoning 3d segmentation via large language model. In International Conference on 3D Vision 2025 , 2025. 7
- [17] Pin-Hao Huang, Han-Hung Lee, Hwann-Tzong Chen, and Tyng-Luh Liu. Text-guided graph neural networks for referring 3d instance segmentation. In AAAI , 2021. 2, 3, 7
- [18] Shijia Huang, Yilun Chen, Jiaya Jia, and Liwei Wang. Multiview transformer for 3d visual grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 15524-15533, 2022. 3
- [19] Jitesh Jain, Jiachen Li, Mang Tik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 2989-2998, 2023. 3
- [20] Nikhil Keetha, Norman M¨ uller, Johannes Sch¨ onberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fischer, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, et al. Mapanything: Universal feed-forward metric 3d reconstruction. arXiv preprint arXiv:2509.13414 , 2025. 3
- [21] Yongmin Kim, Chenhui Chu, and Sadao Kurohashi. Flexible visual grounding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop , pages 285-299, 2022. 3
- [22] Vincent Leroy, Yohann Cabon, and J´ erˆ ome Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision , pages 71-91. Springer, 2024. 3
- [23] Hongyang Li, Jinyuan Qu, and Lei Zhang. Ovseg3r: Learn open-vocabulary instance segmentation from 2d via 3d reconstruction. arXiv preprint arXiv:2509.23541 , 2025. 3
- [24] Yicong Li, Xiang Wang, Junbin Xiao, Wei Ji, and TatSeng Chua. Transformer-empowered invariant grounding

- for video question answering. IEEE Transactions on Pattern Analysis and Machine Intelligence , 2023. 3
- [25] Tianming Liang, Kun-Yu Lin, Chaolei Tan, Jianguo Zhang, Wei-Shi Zheng, and Jian-Fang Hu. Referdino: Referring video object segmentation with visual grounding foundations. arXiv preprint arXiv:2501.14607 , 2025. 6
- [26] Haojia Lin, Yongdong Luo, Xiawu Zheng, Lijiang Li, Fei Chao, Taisong Jin, Donghao Luo, Yan Wang, Liujuan Cao, and Rongrong Ji. A unified framework for 3d point cloud visual grounding. arXiv:2308.11887 , 2023. 3
- [27] Xuexun Liu, Xiaoxu Xu, Jinlong Li, Qiudan Zhang, Xu Wang, Nicu Sebe, and Lin Ma. Less: Label-efficient and single-stage referring 3d segmentation. arXiv:2410.13294 , 2024. 3, 6, 7
- [28] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 , 2019. 6
- [29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 , 2017. 6
- [30] Junyu Luo, Jiahui Fu, Xianghao Kong, Chen Gao, Haibing Ren, Hao Shen, Huaxia Xia, and Si Liu. 3d-sps: Single-stage 3d visual grounding via referred point progressive selection. In CVPR , 2022. 3
- [31] Riku Murai, Eric Dexheimer, and Andrew J Davison. Mast3r-slam: Real-time dense slam with 3d reconstruction priors. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 16695-16705, 2025. 3
- [32] Zador Pataki, Paul-Edouard Sarlin, Johannes L Sch¨ onberger, and Marc Pollefeys. Mp-sfm: Monocular surface priors for robust structure-from-motion. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 21891-21901, 2025. 3
- [33] Charles R Qi, Or Litany, Kaiming He, and Leonidas J Guibas. Deep hough voting for 3d object detection in point clouds. In proceedings of the IEEE/CVF International Conference on Computer Vision , pages 9277-9286, 2019. 3
- [34] Zhipeng Qian, Yiwei Ma, Jiayi Ji, and Xiaoshuai Sun. Xrefseg3d: Enhancing referring 3d instance segmentation via structured cross-modal graph neural networks. In AAAI , 2024. 3
- [35] You Shen, Zhipeng Zhang, Yansong Qu, and Liujuan Cao. Fastvggt: Training-free acceleration of visual geometry transformer. arXiv preprint arXiv:2509.02560 , 2025. 3
- [36] Sanjay Subramanian, William Merrill, Trevor Darrell, Matt Gardner, Sameer Singh, and Anna Rohrbach. Reclip: A strong zero-shot baseline for referring expression comprehension. arXiv preprint arXiv:2204.05991 , 2022. 3
- [37] Grounded Transformer. Iggt: Instance-grounded geometry trans-former for semantic 3d reconstruction. 3
- [38] Chung-Shien Brian Wang, Christian Schmidt, Jens Piekenbrinck, and Bastian Leibe. Faster vggt with block-sparse global attention. arXiv preprint arXiv:2509.07120 , 2025. 3
- [39] Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. arXiv preprint arXiv:2408.16061 , 2024. 3
- [40] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 5294-5306, 2025. 3
- [41] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings of the Computer Vision and Pattern Recognition Conference , pages 10510-10522, 2025. 3
- [42] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 2069720709, 2024. 3
- [43] Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. Pi3: Permutation-equivariant visual geometry learning. arXiv preprint arXiv:2507.13347 , 2025. 3, 6
- [44] Yi Ru Wang, Yuchi Zhao, Haoping Xu, Saggi Eppel, Al´ an Aspuru-Guzik, Florian Shkurti, and Animesh Garg. Mvtrans: Multi-view perception of transparent objects. arXiv preprint arXiv:2302.11683 , 2023. 3
- [45] WinT3R WinT3R. Wint3r: Window-based streaming reconstruction with camera token pool. 3
- [46] Changli Wu, Jiayi Ji, Haowei Wang, Yiwei Ma, You Huang, Gen Luo, Hao Fei, Xiaoshuai Sun, Rongrong Ji, et al. Rgsan: Rule-guided spatial awareness network for end-to-end 3d referring expression segmentation. Advances in Neural Information Processing Systems , 37:110972-110999, 2024. 2, 3, 7
- [47] Changli Wu, Yihang Liu, Jiayi Ji, Yiwei Ma, Haowei Wang, Gen Luo, Henghui Ding, Xiaoshuai Sun, and Rongrong Ji. 3d-gres: Generalized 3d referring expression segmentation. In ACM MM , 2024.
- [48] Changli Wu, Yiwei Ma, Qi Chen, Haowei Wang, Gen Luo, Jiayi Ji, and Xiaoshuai Sun. 3d-stmn: Dependency-driven superpoint-text matching network for end-to-end 3d referring expression segmentation. In AAAI , 2024. 2, 3, 7
- [49] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. In Forty-first International Conference on Machine Learning , 2024. 3
- [50] Qi Xu, Dongxu Wei, Lingzhe Zhao, Wenpu Li, Zhangchi Huang, Shunping Ji, and Peidong Liu. Siu3r: Simultaneous scene understanding and 3d reconstruction beyond feature alignment. arXiv preprint arXiv:2507.02705 , 2025. 3
- [51] Zhihao Yuan, Xu Yan, Yinghong Liao, Ruimao Zhang, Sheng Wang, Zhen Li, and Shuguang Cui. Instancerefer: Cooperative holistic understanding for visual grounding on point clouds through instance multi-level contextual referring. In ICCV , 2021. 3
- [52] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision , pages 3836-3847, 2023. 5

- [53] Yiming Zhang, ZeMing Gong, and Angel X Chang. Multi3drefer: Grounding text description to multiple 3d objects. In ICCV , 2023. 2
- [54] Zhi Zhang, Helen Yannakoudakis, Xiantong Zhen, and Ekaterina Shutova. Ck-transformer: Commonsense knowledge enhanced transformers for referring expression comprehension. arXiv preprint arXiv:2302.09027 , 2023. 3
- [55] Lichen Zhao, Daigang Cai, Lu Sheng, and Dong Xu. 3dvgtransformer: Relation modeling for visual grounding on point clouds. In ICCV , 2021. 2