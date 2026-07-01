# Harmonic RoPE: 面向符号音乐生成的音程感知旋转位置编码

## 1. 核心假设

本项目的一个基本假设是：

> 音乐听感的最小关系单元不是单个音符，而是音程。

单个音符本身并不直接构成音乐意义。真正产生稳定听感、功能倾向和结构关系的，往往是两个或多个音之间的相对关系。例如：

* 八度关系带来高度相似性；
* 五度关系带来强烈的调性/功能联系；
* 三度关系塑造和声色彩；
* 三全音关系带来不稳定性和解决倾向；
* 同一音程在不同时间位置上，可能分别表现为和声音程、旋律跳进、经过音、延留音或模进的一部分。

因此，符号音乐模型如果只把音符当作离散 token 学习，很容易把音高关系退化成共现统计，而难以直接看到音程背后的结构。

Harmonic RoPE 的目标是：
**让 Transformer 的注意力机制直接感知音程关系，而不仅仅是记住音符 token 之间的共现模式。**

---

## 2. 为什么音程天然适合 RoPE？

### 2.1 音高可以表示为对数频率

设一个音的频率为 $f$。在音乐中，两个音之间的音程主要由频率比决定：

$$r = \frac{f_2}{f_1}$$

如果取对数，则音程可以写成：

$$\Delta p = \log \frac{f_2}{f_1}$$

这意味着：

$$\log f_2 - \log f_1 = \log \frac{f_2}{f_1}$$

也就是说，**音程本质上是对数频率空间中的差值**。

这和普通 RoPE 的基本思想非常接近：
RoPE 并不是直接编码绝对位置，而是让注意力内积天然依赖两个 token 之间的相对位置差。

---

### 2.2 八度等价与商群结构

在音乐听感中，频率相差一倍的两个音具有强烈相似性：

$$f \sim 2f$$

对应到对数频率空间：

$$\log(2f) = \log f + \log 2$$

因此，如果把八度看作一个周期，那么音高空间可以近似理解为：

$$\mathbb{R} / \log 2\mathbb{Z}$$

也就是：
**对数频率空间按八度周期取商。**

在十二平均律中，如果用半音编号表示音高，例如 MIDI pitch $p$，那么八度等价可以写成：

$$p \sim p + 12$$

音级空间可以近似表示为：

$$\mathbb{Z} / 12\mathbb{Z}$$

这就是音程的商群特质：
相差一个或多个周期的音，在某些音乐功能上可以被视为等价或相近。

---

### 2.3 五度圈也是一种周期结构

除了八度，五度也具有重要的音乐功能意义。
在十二平均律中，一个纯五度近似等于 7 个半音。因此沿五度圈移动，可以写成：

$$p \mapsto p + 7 \pmod{12}$$

连续五度移动会遍历全部 12 个 pitch class：

$$0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5$$

五度圈不是简单的音高距离，而更接近一种调性距离、功能距离或和声邻近关系。

因此，Harmonic RoPE 可以选择不同周期：

* 以八度为周期，让模型感知 octave equivalence；
* 以五度圈为周期，让模型感知 functional proximity；
* 以更长周期表示线性音高距离与方向性；
* 以多个周期并行表示多尺度音高关系。

---

## 3. RoPE 的基本数学形式

标准 RoPE 会把 hidden state 的相邻两维看成一个二维向量：

$$\mathbf{x}_j =
\begin{bmatrix}
x_{2j} \\
x_{2j+1}
\end{bmatrix}$$

然后根据位置 $m$，对这个二维向量做旋转：

$$R(m\theta_j)\mathbf{x}_j$$

其中旋转矩阵为：

$$R(\phi) =
\begin{bmatrix}
\cos \phi & -\sin \phi \\
\sin \phi & \cos \phi
\end{bmatrix}$$

所以：

$$R(m\theta_j) =
\begin{bmatrix}
\cos(m\theta_j) & -\sin(m\theta_j) \\
\sin(m\theta_j) & \cos(m\theta_j)
\end{bmatrix}$$

如果两个 token 的位置分别是 $m$ 和 $n$，那么它们的 query/key 内积会包含相对位置差：

$$R(m\theta_j)^\top R(n\theta_j) = R((n-m)\theta_j)$$

也就是说，注意力机制看到的不是单纯的绝对位置，而是：

$$n - m$$

这正是 RoPE 的核心性质。

---

## 4. 从 Temporal RoPE 到 Harmonic RoPE

### 4.1 Temporal RoPE

在普通语言模型中，位置通常是线性 token index：

$$m = 0, 1, 2, 3, \dots$$

但在符号音乐中，线性 token 顺序并不等于真实音乐时间。

例如，以下 token 在线性序列中可能相隔较远：

```text
BAR_1 BEAT_1 V1 C4 DUR_1/4 V2 E3 DUR_1/4 V3 G3 DUR_1/4

```

但从音乐时间上看，它们可能同时发生。

因此，Temporal RoPE 不应只使用 token index，而应使用音乐时间坐标：

$$t \in \mathbb{Q}$$

例如：

$$t = \text{bar offset} + \text{beat offset}$$

或者使用更细的 rational time / fractional time：

$$t = \frac{k}{64}$$

Temporal RoPE 的相位可以写成：

$$\phi_t = \omega_t t$$

其中 $\omega_t$ 是该旋转维度上的时间频率。

Temporal RoPE 的目标是让模型知道：

* 哪些音符同时发生；
* 哪些音符相邻；
* 哪些音符在同一拍；
* 哪些音符跨拍延续；
* 哪些音符处于远距离呼应关系。

---

### 4.2 Harmonic RoPE

Harmonic RoPE 则把位置从“时间坐标”扩展到“音高/音程坐标”。

设音高坐标为 $h$。
最简单情况下，可以令：

$$h = p$$

其中 $p$ 是 MIDI pitch 或半音编号。

那么 Harmonic RoPE 的相位可以写成：

$$\phi_h = \omega_h h$$

两个音之间的相对 harmonic phase 为：

$$\Delta \phi_h = \omega_h(h_2 - h_1)$$

由于 $h_2 - h_1$ 就是半音意义上的音程，因此 Harmonic RoPE 可以让注意力机制直接看到音程关系。

如果选择周期为八度，则可以令：

$$\omega_{\text{oct}} = \frac{2\pi}{12}$$

此时相差 12 个半音的两个音会旋转一整圈：

$$\omega_{\text{oct}}(h+12) = \omega_{\text{oct}}h + 2\pi$$

也就是说：

$$R(\omega_{\text{oct}}(h+12)) = R(\omega_{\text{oct}}h)$$

这让模型直接看到八度等价。

如果选择五度圈坐标，可以先定义：

$$c(p) = 7^{-1}p \pmod{12}$$

或者直接使用预定义的 circle-of-fifths index。然后令：

$$\phi_{\text{fifth}} = \omega_{\text{fifth}} c(p)$$

这样模型看到的不是线性半音距离，而是五度圈上的距离。

---

## 5. 纯 Harmonic RoPE 的问题：容易退化成查表

纯 Harmonic RoPE 的形式是：

$$\phi_h = \omega_h h$$

两个 token 之间的相对相位是：

$$\Delta \phi_h = \omega_h(h_2 - h_1)$$

这使得模型可以看到音程，但它也带来一个问题：

> 纯 Harmonic RoPE 只知道两个音之间是什么音程，却不知道这个音程发生在什么时间关系中。

例如，以下几种情况在纯 harmonic phase 中可能具有相同的音程差：

1. 同一时刻发声的 C-G；
2. 旋律中相邻两个音 C-G；
3. 隔一小节呼应的 C-G；
4. 五度模进中的 C-G；
5. 延留音解决过程中的 C-G。

它们的 harmonic difference 都可能是：

$$\Delta h = 7$$

但它们在音乐意义上完全不同。

更进一步，Harmonic RoPE 的旋转速度通常比 Temporal RoPE 快得多。

Temporal RoPE 的位置 $t$ 往往缓慢增加，例如：

$$0, \frac{1}{4}, \frac{1}{2}, 1, 2, 3, \dots$$

而 harmonic coordinate 可能在相邻音符之间快速变化：

$$C4 \rightarrow G5 \rightarrow F3 \rightarrow B4$$

如果 harmonic 频率设计较高，旋转角可能轻易跨越多个周期。
这时纯 Harmonic RoPE 不再像连续几何空间，而更像一种周期性哈希：

$$h \mapsto R(\omega_h h)$$

模型看到的是一组高度周期化的音程类别。
这在一定程度上有用，但它很容易退化为：

> interval lookup table

也就是说，模型只是学会“某类音程常见或不常见”，而不是理解“这个音程如何在时间中发挥功能”。

---

## 6. 核心创新：Temporal-Harmonic Coupled RoPE

为了解决纯 Harmonic RoPE 过度静态的问题，本项目引入 Temporal-Harmonic Coupled RoPE。

基本形式是把时间相位和 harmonic 相位相加：

$$\phi_{th} = \phi_t + \phi_h$$

也就是：

$$\phi_{th} = \omega_t t + \omega_h h$$

两个 token 之间的相对相位为：

$$\Delta \phi_{th} = \omega_t(t_2 - t_1) + \omega_h(h_2 - h_1)$$

记作：

$$\Delta \phi_{th} = \omega_t \Delta t + \omega_h \Delta h$$

这意味着模型不再只看到“音程是什么”，而是看到：

> 某个音程在某个时间距离中的表现。

例如，同样是 C-G：

* 如果两个音同时发生（$\Delta t = 0$），则：

$$\Delta \phi_{th} = \omega_h \Delta h$$

* 如果两个音相隔一个八分音符（$\Delta t = \frac{1}{2}$），则：

$$\Delta \phi_{th} = \omega_t \cdot \frac{1}{2} + \omega_h \Delta h$$

* 如果两个音相隔一小节（$\Delta t = 1 \text{ bar}$），则：

$$\Delta \phi_{th} = \omega_t \cdot 1 + \omega_h \Delta h$$

因此，同一个音程在不同时间距离中会形成不同的旋转关系。

这使得模型可以区分：

* 同时发生的和声音程；
* 旋律中的跳进；
* 对位中的声部运动；
* 延留与解决；
* 模进中的音高迁移；
* 远距离主题呼应。

---

## 7. 为什么时间相位是重要的修正项？

Temporal RoPE 的旋转速度通常较慢。
它不会在短时间内快速跨越多个周期，而是提供一种稳定、平滑的时间坐标。

Harmonic RoPE 的旋转速度则可以很快。
因为音高可以在短时间内跳动很大，且八度、五度等周期本身较短。

如果只使用 Harmonic RoPE：

$$\phi_h = \omega_h h$$

模型可能只看到一个静态音程类别。

但加入时间相位后：

$$\phi_{th} = \omega_h h + \omega_t t$$

其中 $\omega_t t$ 虽然变化较慢，却会对 harmonic phase 产生持续修正。

这个修正项的意义是：

> 时间相位把静态音程变成动态音程，把音程类别变成时间中的音乐关系。

从注意力角度看，两个 token 的相对关系从 $\omega_h \Delta h$ 变成了：

$$\omega_h \Delta h + \omega_t \Delta t$$

这意味着同一个 $\Delta h$ 不再对应唯一的注意力模式。
它还必须结合 $\Delta t$ 才能决定最终关系。

因此，Temporal-Harmonic Coupled RoPE 可以缓解纯 harmonic phase 的查表化问题。

---

## 8. 为什么相位相加在数学上成立？

旋转矩阵有一个重要性质：

$$R(a)R(b) = R(a+b)$$

证明如下。设：

$$R(a) =
\begin{bmatrix}
\cos a & -\sin a \\
\sin a & \cos a
\end{bmatrix}$$

$$R(b) =
\begin{bmatrix}
\cos b & -\sin b \\
\sin b & \cos b
\end{bmatrix}$$

则：

$$R(a)R(b) =
\begin{bmatrix}
\cos a\cos b - \sin a\sin b & -\cos a\sin b - \sin a\cos b \\
\sin a\cos b + \cos a\sin b & -\sin a\sin b + \cos a\cos b
\end{bmatrix}$$

根据三角恒等式：

$$\cos(a+b) = \cos a\cos b - \sin a\sin b$$

$$\sin(a+b) = \sin a\cos b + \cos a\sin b$$

因此：

$$R(a)R(b) =
\begin{bmatrix}
\cos(a+b) & -\sin(a+b) \\
\sin(a+b) & \cos(a+b)
\end{bmatrix}
= R(a+b)$$

所以，如果时间旋转和 harmonic 旋转作用在同一个二维子空间中，那么先做时间旋转再做 harmonic 旋转，等价于直接做一次合并旋转：

$$R(\phi_t)R(\phi_h) = R(\phi_t + \phi_h)$$

这使得 Temporal-Harmonic Coupled RoPE 在工程上也非常简洁：
不需要额外 attention bias，也不需要修改 attention kernel，只需要改变 rotary phase 的构造方式。

---

## 9. 注意力内积如何看到相对时间-音程关系？

设第 $i$ 个 token 的时间和 harmonic 坐标为 $(t_i, h_i)$，第 $j$ 个 token 的时间和 harmonic 坐标为 $(t_j, h_j)$。

它们对应的 coupled phase 为：

$$\phi_i = \omega_t t_i + \omega_h h_i$$

$$\phi_j = \omega_t t_j + \omega_h h_j$$

注意力中 query/key 的旋转关系包含：

$$R(\phi_i)^\top R(\phi_j)$$

由于旋转矩阵满足 $R(\phi_i)^\top = R(-\phi_i)$，因此：

$$R(\phi_i)^\top R(\phi_j) = R(-\phi_i)R(\phi_j) = R(\phi_j - \phi_i)$$

代入 $\phi_i$ 和 $\phi_j$：

$$\phi_j - \phi_i = \omega_t(t_j - t_i) + \omega_h(h_j - h_i)$$

即：

$$\Delta \phi = \omega_t \Delta t + \omega_h \Delta h$$

所以 Coupled RoPE 让 attention score 直接依赖 $(\Delta t, \Delta h)$，而不是只依赖 token index 或只依赖静态音程。

这正是它的核心作用。

---

## 10. 为什么仍然保留少量纯 Harmonic RoPE？

虽然纯 Harmonic RoPE 容易退化成查表，但它并不是完全无用。

音乐中确实存在一些相对稳定、与时间无关的音高关系：

* C 和 G 在五度关系上接近；
* C4 和 C5 具有八度等价性；
* 三度关系通常比三全音关系更稳定；
* 相差极远的音往往关系更弱；
* 上行音程和下行音程在旋律感知上并不完全等价。

因此，本项目保留少量纯 Harmonic RoPE 维度，用于提供稳定的音高空间先验。

但这部分不再使用短周期、高速旋转为主，而是更偏向低频、长周期、线性方向感知。

---

## 11. 长周期 Harmonic RoPE：保留线性音高距离与方向性

短周期 Harmonic RoPE 适合表达八度等价或五度圈关系，但它会弱化线性距离。

例如，如果只使用八度周期：

$$C3 \sim C4 \sim C5 \sim C6$$

模型会很容易看到它们的相似性，但可能不容易知道 $C3$ 和 $C6$ 在真实音域上相距很远。

因此，需要少量长周期 harmonic 维度来表达：

* 音高上行/下行方向；
* 音域距离；
* register difference；
* 远距离音高关系的衰减；
* 极高音区与极低音区的不同功能。

可以设一个覆盖常见音域的大周期，例如 $P_{\text{range}} = 96$，表示 8 个八度左右的范围。对应角频率为：

$$\omega_{\text{range}} = \frac{2\pi}{P_{\text{range}}}$$

纯 long-range harmonic phase 为：

$$\phi_{\text{range}} = \omega_{\text{range}} h$$

这样，相差一个八度的两个音不会完全重合，而只会产生较小角度差：

$$\Delta \phi_{\text{range}} = \omega_{\text{range}} \cdot 12 = \frac{2\pi}{96} \cdot 12 = \frac{\pi}{4}$$

而相差 4 个八度的音会产生更大差异：

$$\Delta \phi_{\text{range}} = \frac{2\pi}{96} \cdot 48 = \pi$$

这种设计可以帮助模型看到：

> 八度等价并不意味着所有八度距离都完全相同。音区距离本身也具有音乐意义。

---

## 12. 最终设计：Factorized + Coupled Harmonic RoPE

本项目采用混合式设计，而不是把所有维度都做同一种旋转。

整体 rotary subspace 可以分为四类：

```text
1. Temporal-only subspace
2. Coupled temporal-harmonic subspace
3. Pure harmonic long-range subspace
4. Identity/content subspace

```

### 12.1 Temporal-only subspace

用于表达纯时间关系：

$$\phi_t = \omega_t t$$

功能：

* 同时性；
* 相邻性；
* 拍点关系；
* 小节关系；
* 长程时间距离。

这是符号音乐建模的主干位置编码。

---

### 12.2 Coupled temporal-harmonic subspace

用于表达时间中的音程关系：

$$\phi_{th} = \omega_t t + \omega_h h$$

功能：

* 和声音程 vs 旋律音程；
* 模进；
* 对位运动；
* 延留与解决；
* 时间中的和声推进；
* 同一音程在不同时间距离中的不同功能。

这是 Harmonic RoPE 的核心创新部分。

---

### 12.3 Pure harmonic long-range subspace

用于表达音域距离和线性方向：

$$\phi_{\text{range}} = \omega_{\text{range}}h$$

其中 $P_{\text{range}}$ 较大，例如覆盖 7–8 个八度：

$$\omega_{\text{range}} = \frac{2\pi}{P_{\text{range}}}$$

功能：

* 音高方向；
* 音域距离；
* 高低音区差异；
* 远距离音高关系衰减。

这部分不加时间修正，因为它表达的是稳定的线性音高空间，而不是时间中的音程功能。

---

### 12.4 Identity/content subspace

保留部分不做旋转的维度：

$$\phi = 0$$

功能：

* 保留 token 内容表达；
* 避免所有维度都被位置结构强约束；
* 让模型自由学习非几何化特征。

---

## 13. 实现建议

### 13.1 输入坐标

Harmonic RoPE 的设计并不要求输入必须是离散 MIDI pitch。MIDI 编号很适合作为早期实现配置，因为它简单、稳定、方便与现有 MusicXML / MIDI / symbolic dataset 对齐。但从设计初衷上，Harmonic RoPE 更适合被理解为一种基于真实频率或对数频率的连续坐标编码。

也就是说，一个音高 token 可以附带的核心坐标不是必须为：

`pitch_id = MIDI pitch`

而可以更一般地表示为：

```text
frequency_hz        # 真实频率，例如 440.0 Hz
log_freq_coord      # 对数频率坐标，例如 log2(f / f_ref)
pitch_coord         # 可选：由 log_freq_coord 映射得到的连续或离散音高坐标

```

在十二平均律中，MIDI pitch 可以看作对数频率坐标的一种离散化形式。若以 A4 = 440 Hz，对应 MIDI 69，则有：

$$p = 69 + 12 \log_2 \frac{f}{440}$$

当 $f$ 恰好来自十二平均律时，$p$ 通常是整数；当输入来自真实演奏、微分音、非十二平均律或连续频率估计时，$p$ 可以是小数。

因此，初期实现可以使用 MIDI pitch，但模型结构本身不应依赖 MIDI 的离散性。

在默认的十二平均律配置下，每个 token 可以附带以下坐标：

```text
time_id             # rational / fractional musical time
frequency_hz        # optional: real frequency, if available
log_freq_coord      # log2(f / f_ref), continuous pitch coordinate
pitch_id            # optional: MIDI pitch or MIDI-like continuous pitch
pitch_class_id      # optional: pitch_id % 12, for 12-TET only
fifth_id            # optional: circle-of-fifths index, for 12-TET functional space
voice_id            # voice / part index
token_type          # NOTE, DUR, BAR, BEAT, SUST, EXT, etc.

```

其中：

* `pitch_id`
* `pitch_class_id`
* `fifth_id`

都应被视为十二平均律下的便捷实现，而不是 Harmonic RoPE 的理论前提。

现代西方音乐的大部分符号数据默认使用十二平均律。十二平均律的一个重要特点是具有平移不变性：任意旋律或和声结构整体移调后，其音程结构保持不变。因此，使用：

* `pitch_id % 12`
* `circle-of-fifths index`

可以很好地服务于常见调性音乐建模。

但音乐系统并不总是十二平均律。历史音乐、民族音乐、现代音乐、实验音乐和微分音音乐中都可能出现：

* 非 12 音划分
* 非等分音阶
* 非对称律制
* 不具备简单移调不变性的音高系统
* 真实频率偏移
* 演奏 intonation drift

因此，更通用的实现方式应允许用户自定义 harmonic coordinate mapping：

`frequency_hz / log_freq_coord -> harmonic_coord`

例如：

```text
12-TET:
    harmonic_coord = MIDI-like pitch coordinate

24-TET:
    harmonic_coord = 24 * log2(f / f_ref)

just intonation:
    harmonic_coord = ratio-based or prime-factor coordinate

custom tuning:
    harmonic_coord = user-defined pitch-space coordinate

```

这样 Harmonic RoPE 不依赖某一种固定律制，而是可以复用到不同 tuning system 中。

对于非音高 token，可以采用：

1. harmonic phase = 0
2. inherit previous pitch coordinate
3. inherit current beat/chord coordinate

初期建议使用最简单稳定的策略：

```text
non-pitch token: harmonic phase = 0

```

这样可以避免引入复杂状态传播，也能保证不同 token 类型之间的行为足够可控。

### 13.2 Phase construction

Harmonic RoPE 的 phase construction 可以从连续频率坐标出发，再根据当前配置映射到不同 harmonic subspace。

伪代码如下：

```python
def build_music_rope_phase(
    time_coord,
    frequency_hz=None,
    log_freq_coord=None,
    pitch_coord=None,
    pitch_class_coord=None,
    fifth_coord=None,
    tuning_config=None,
    rotary_dim=None,
):
    phase = zeros(rotary_dim // 2)

    # ------------------------------------------------------------
    # 0. Build pitch / harmonic coordinates
    # ------------------------------------------------------------
    # The model does not require discrete MIDI pitch.
    # MIDI-like pitch is only one possible coordinate system.
    if log_freq_coord is None and frequency_hz is not None:
        log_freq_coord = log2(frequency_hz / tuning_config.f_ref)

    if pitch_coord is None and log_freq_coord is not None:
        # In 12-TET, this is equivalent to continuous MIDI-like pitch.
        pitch_coord = tuning_config.pitch_scale * log_freq_coord + tuning_config.pitch_offset

    if pitch_class_coord is None and tuning_config.use_pitch_class:
        # Usually only valid for 12-TET or other explicitly periodic systems.
        pitch_class_coord = pitch_coord % tuning_config.period

    if fifth_coord is None and tuning_config.use_fifths:
        # Usually only valid for 12-TET or a defined functional lattice.
        fifth_coord = tuning_config.to_fifth_coord(pitch_class_coord)

    # ------------------------------------------------------------
    # 1. temporal-only subspace
    # ------------------------------------------------------------
    phase[temp_slice] = omega_time * time_coord

    # ------------------------------------------------------------
    # 2. coupled temporal-harmonic subspace
    # ------------------------------------------------------------
    # This subspace models interval function in time.
    # In 12-TET, fifth_coord or pitch_class_coord can be used.
    # In other tuning systems, this can be replaced by any user-defined
    # harmonic coordinate.
    phase[coupled_slice] = (
        omega_time_coupled * time_coord
        + omega_harmonic * tuning_config.harmonic_coord(
            pitch_coord=pitch_coord,
            pitch_class_coord=pitch_class_coord,
            fifth_coord=fifth_coord,
            log_freq_coord=log_freq_coord,
        )
    )

    # ------------------------------------------------------------
    # 3. pure long-range harmonic subspace
    # ------------------------------------------------------------
    # This subspace preserves pitch direction and register distance.
    # It should usually use continuous pitch/log-frequency coordinates,
    # not pitch class.
    phase[range_slice] = omega_range * pitch_coord

    # ------------------------------------------------------------
    # 4. identity/content subspace
    # ------------------------------------------------------------
    phase[identity_slice] = 0.0

    return phase

```

其中：

* `omega_time` 使用类似普通 RoPE 的多频率设计；
* `omega_harmonic` 可以使用八度周期、五度圈周期或用户自定义 tuning 周期；
* `omega_range` 使用较低频率，覆盖常见音域，用于保留音高方向与 register distance；
* `identity_slice` 不旋转，用于保留普通 token 内容表达；
* `tuning_config` 负责把真实频率、对数频率或 MIDI-like pitch 映射到当前律制下的 harmonic coordinate。

在十二平均律中，可以使用：

```python
pitch_coord = MIDI-like pitch
pitch_class_coord = pitch_coord % 12
fifth_coord = circle_of_fifths[pitch_class_coord]

```

在非十二平均律中，则不应强行使用 `pitch_id % 12`，而应改为：

```python
harmonic_coord = tuning_config.harmonic_coord(log_freq_coord)

```

这样可以保证 Harmonic RoPE 在不同律制下仍然复用同一套 phase construction 框架。

### 13.3 KV cache

推理时，应在写入 KV cache 前完成旋转：

```python
q = apply_rope(q, phase)
k = apply_rope(k, phase)

kv_cache.append(k, v)

```

这样历史 key 不需要在每一步重新计算 harmonic phase。

如果 harmonic coordinate 是离散的，例如十二平均律 MIDI / pitch class / fifth index，则可以预计算 phase cache：

```text
time_coord -> temporal phase
pitch_class -> octave phase
fifth_id -> fifth phase
pitch_id -> range phase

```

推理时只需要查表并相加。

如果输入是连续频率或非十二平均律坐标，则 phase 不一定能完全离散查表。此时可以采用两种方式：

1. 直接根据 `log_freq_coord` 实时计算 phase
2. 对连续频率坐标做高精度量化后缓存 phase

例如：

```text
log_freq_coord -> continuous harmonic phase
quantized_log_freq_coord -> cached harmonic phase

```

因此，phase cache 是一种工程优化，而不是 Harmonic RoPE 的理论前提。

Harmonic RoPE 的核心只要求每个音高事件能够映射到某种 harmonic coordinate；这个 coordinate 可以来自 MIDI，也可以来自真实频率、连续音高估计或任意自定义律制。

---

## 14. 预期收益

Harmonic RoPE 不一定会显著降低整体 validation loss，但它预期会改善结构性错误，减少模型从统计规律中找到音乐规则的成本、让物理声学归纳偏置对模型清晰可见：

* 更稳定地区分同时和声与旋律跳进；
* 更好捕捉五度模进与序列结构；
* 更少出现不合理的远距离音高跳动；
* 更好处理跨声部音程关系；
* 更好泛化到不同调性；
* 更好保持局部改写时的和声兼容性；
* 更少把音乐时间关系误认为线性 token 距离。

尤其在小模型和数据有限场景下，显式归纳偏置可以减少模型从数据中自行学习音乐结构的难度。

---

## 15. 总结

Harmonic RoPE 的核心思想是：

> 音乐不是由孤立音符构成，而是由时间中的音程关系构成。

音程在对数频率空间中具有明显的周期性和商群结构。
这种周期性与 RoPE 的二维子空间旋转天然匹配。

纯 Harmonic RoPE 可以让模型看到音程，但它容易退化成静态音程查表。
因此引入 Temporal-Harmonic Coupled RoPE：

$$\phi_{th} = \omega_t t + \omega_h h$$

它让模型看到：

$$\Delta \phi_{th} = \omega_t \Delta t + \omega_h \Delta h$$

也就是：

> 某个音程在某个时间距离中的功能。

同时，模型仍然保留少量纯 harmonic long-range subspace，用于表达音域距离、线性方向和 register difference。

最终设计可以概括为：

```text
Temporal-only:                 音乐时间主干
Coupled temporal-harmonic:      时间中的音程功能
Pure harmonic long-range:       音高方向与音域距离
Identity/content:               普通内容表达

```

这使模型不再只把符号音乐当作线性 token 序列，而是能够在注意力层面直接感知音乐中最核心的结构关系：
**时间中的音程。**
