# GPT-2 attention：修正後完整實驗結果

正式分析入口：[General_Analysis_GPT2.ipynb](../../General_Analysis_GPT2.ipynb)。
已執行、有圖與數值的副本：[General_Analysis_GPT2_executed.ipynb](General_Analysis_GPT2_executed.ipynb)。
本文的 layer/head 編號均從 1 開始。除特別標示外，下列數值來自 **842 篇 validation articles**，先在各篇內平均，再讓文章等權。

## 最重要的修正

先前 158 篇的程式把 `\n\n` 分開 tokenize，人工固定成 ID 628 (`ĊĊ`)。
稽核發現全部 1000 筆舊序列都不等於同一段文字的正常 GPT-2 tokenization。
修正為把兩段文字接起來一起 tokenize 後，這批資料的自然段落邊界均為兩個 ID 198 (`Ċ`, `Ċ`)。
因此撤回舊報告中「自然 `ĊĊ` sink」的說法，舊結果僅保留作為稽核紀錄。
本次沿用相同 1000 篇文章及段落選擇，不因結果更换樣本。

## 與原論文／notebook 的對應

參考 Clark et al., [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341)，Sections 3.2–3.3，以及 repo 原始 `General_Analysis.ipynb`。

| 原圖／研究問題 | 本次實作與必要差異 |
|---|---|
| Figure 1：有特徵的 head attention 連線 | 同一篇、同一個段落邊界視窗，並列標點／邊界／首位置／前一 token／broad heads；使用原 attention 權重，不對裁切視窗重新正規化，標示保留下來的 mass |
| Figure 2：各 head 對特殊 token／標點的偏好 | 沿用逐層散點與 layer mean，觀察 initial EOT、自然換行邊界、句點／逗點，並保留原 notebook 的 relative-position panel |
| Separator queries 與其他 queries | 比較 boundary→boundary 與 other→boundary；另只取已能看見邊界的 queries，避免把 GPT-2 無法看見未來造成的零值當作不偏好 |
| Figure 4：focused vs broad | 沿用上下兩面板 raw entropy 與 uniform reference；GPT-2 reference 必須是每個 query 的 `log(q+1)`，不能直接用全文長度 |
| `[CLS]` query entropy | 改呈現 trailing EOT 作為特殊 query 的診斷，**不宣稱與 CLS 等價**；GPT-2 的首 token 只能看自己，entropy 必然為 0，不能直接代換 CLS |
| 原文的 no-op 假說 | 這次沒有重做原文的 gradient-based 證據；attention enrichment、低 entropy 或統計顯著都不能證明 functional no-op |

繪圖沿用原 notebook 的視覺結構；另用 academic-plotting 的數據圖檢查流程確認標籤、信賴區間與 PNG/PDF 匯出。

## 執行設定與品質檢查

- 主實驗：1000 篇 `<eot> P1 \n\n P2 <eot>`；位置診斷：相同 1000 篇移除 leading EOT，保留 trailing EOT。
- 前 158 篇曾經被檢視，作 discovery；剩餘 842 篇作 held-out validation。八項假設在新模型運算前寫入 [protocol.md](protocol.md)。
- Wikipedia：`wikimedia/wikipedia`，`20231101.en`，revision `ad5752b`，seed 42；每篇只抽一組相鄰段落。
- GPT-2 small：`openai-community/gpt2`，模型與 tokenizer snapshot `607a30d783dfa663caf39e06633721c8d4cfcd7e`。
- 實際運算裝置：本機 NVIDIA GeForce RTX 3080 Laptop GPU，並非 nano4；兩組 inference＋統計約 535 秒。
- 主實驗長度 70–1023，median 165.5；最大設定 1024，不把短段落補到固定長度。ordinary query 共 194,091 個。
- 1000 筆全部通過 canonical tokenization round-trip、雙換行、只有頭尾 EOT、長度與不同文章檢查。
- 固定原文章／段落重新 preprocessing 的兩份 JSONL **byte-for-byte 一致**，SHA-256：`b2284494eea86fb9a9c02d178bf712313a7bac4540daa11fc973aaf372f60e0b`。此項檢查不是重新下載整個 Wikipedia 的抽樣重現測試。
- 兩組最大 attention row-sum 誤差分別為 `4.1795e-4`、`4.2880e-4`；future attention 均為 0。統計前只在 causal support 重新正規化，移除 float16 捨入誤差。
- streaming：一筆 dense map 算完統計便釋放，不累積 dense 檔案；兩組 per-document NPZ 約 93.7 MiB，少量示例 head 約 111 KiB，連同圖與 notebook 仍遠低於 2 GiB。
- 經使用者授權，刪除舊 158 個 dense `.npy`，釋放 2,142,653,536 bytes；保留 manifest、JSON metadata、舊報告與 CSV。需要舊 maps 時可依原輸入重新計算，沒有保留二進位備份。

完整環境及執行資訊：[stream_manifest.json](stream_manifest.json)。

## Experiment 2：三種不能混為一談的偏好

### 1. 首位置集中，不只是 EOT identity

L08H03 的 ordinary-query attention 有 **92.87%** 指向開頭 EOT，article-bootstrap 95% CI **[92.39%, 93.33%]**。
移除 leading EOT 後，指向第一個正文 token 的 attention 仍為 **92.38% [91.89%, 92.85%]**。

這支持「首位置集中」作為描述，而不是只叫 EOS detector。兩種條件都通過事先固定的 50% 門檻檢驗。
但移除 prefix 同時改變 token identity、序列長度與 absolute positions，因此不能據此隔離各因素的因果作用，也沒有證明兩条件統計等價。

### 2. 標點與自然段落邊界：有 enrichment，不等於佔據大部分 attention

下表排除 self links 及 EOT queries。Distance baseline 在**每一個 query/head** 內保留各相對距離 bucket 的總 attention，再在 bucket keys 間均分；因此同時控制 target 出現次數、query 的可見範圍與粗粒度距離。

| 預先指定 head / target | 實際 mass | Distance-matched mass | Enrichment | 超額 mass 的 95% CI |
|---|---:|---:|---:|---:|
| L03H07 → 句點 | 12.804% | 4.238% | **3.021×** | +8.262 至 +8.870 percentage points |
| L11H04 → 自然段落邊界 | 2.184% | 0.825% | **2.648×** | +1.300 至 +1.420 pp |
| L03H07 → 換行 token | 9.343% | 3.878% | **2.409×** | +5.104 至 +5.841 pp |

這些是明確的內容／邊界偏好，但不能把 2.18% 的 boundary mass 說成與 92.87% 的初始位置 sink 同樣強。
和 BERT 的 separator 分析相比，本次 GPT-2 更顯著的大量集中現象在**初始位置**，不是自然段落邊界。
新的 enrichment 數字也不可與舊 pooled-distance baseline 的倍率直接相減；本次使用更嚴格的 query-matched baseline。

額外描述性診斷（未增加顯著性檢驗）：只在段落邊界後方查詢，L11H04 的 boundary mass 為 4.41%；距離邊界至少 32 positions 時仍為 3.96%。
L03H07 在已有過去句點的 queries 中，argmax 指向任一句點的比例平均 41.98%，指向最近句點約 38.06%；這不是「所有 queries 都指向句點」。

## Experiment 3：低 entropy 至少有兩種不同來源

### 強反例：focused 不等於 self-attention

L05H12 的 causal-normalized entropy 僅 **0.00416**，但 self mass 只有 **0.445%**。
原先「self mass >90%」假設 H5 被否定，不能事後修改門檻當成成功。
它實際約 **98.956% [98.902%, 99.006%]** attention 指向**前一個 token**。
後者是失敗後的描述性解釋，沒有冒稱為另外預先指定且驗證成功的假設。

L08H03 則是另一種 focused head：低 entropy 主要伴隨第一個 key 的大量集中。
在 `q>=32` 的相同 queries 上，原 normalized entropy 為 **0.0725**；只觀察其他 keys 並條件正規化後為 **0.5792**。
這只是對同一組 attention weights 的條件分布統計，不是刪除模型 attention link 或重新執行模型的 ablation。

### 真正較 broad 的 head／layer

- L01H12 normalized entropy：**0.9504 [0.9497, 0.9511]**。
- 在 `q>=32`，平均 **88.25% [87.72%, 88.76%]** queries 的單一最大 attention 不超過 10%；平均最大 attention 為 5.22%。這補足了只報 entropy 的不足。
- Layer 2 平均 normalized entropy **0.7521**，Layer 8 **0.2922**；差值 **0.45994 [0.45831, 0.46152]**。
- 但不能直接推論「深層必然只看少數內容 token」：同樣在 `q>=32`，Layer 8 原 entropy 為 0.2962；條件排除 initial key 後是 0.6099。兩者 support 分別為 `q+1` 與 `q`，使用各自的 uniform normalization。

## 八項固定檢驗：包含失敗結果

CI：以文章為 resampling unit 的 10,000 次 bootstrap，seed 42。
P-value：單尾 exact sign test，檢驗非零文章差值為正的比例是否超過 1/2；八項一起做 Holm correction。
**Sign test 檢驗 median 方向，CI 描述 mean effect，兩者不是同一個 estimand。** Supported 同時要求 adjusted p <0.05 與 mean-effect CI 下界 >0。

| Hypothesis | Mean effect（原 attention 比例單位，H6 為 entropy 差） | 95% CI | Holm p | 結果 |
|---|---:|---:|---:|---|
| H1：L03H07 period − distance baseline | +0.08566 | [0.08262, 0.08870] | 1.68e-213 | 支持 |
| H2：L11H04 boundary − distance baseline | +0.01359 | [0.01300, 0.01420] | 2.46e-181 | 支持 |
| H3：L03H07 newline − distance baseline | +0.05465 | [0.05104, 0.05841] | 2.73e-253 | 支持 |
| H4：L08H03 initial EOT mass − 0.50 | +0.42873 | [0.42392, 0.43329] | 1.44e-250 | 支持 |
| H5：L05H12 self mass − 0.90 | **−0.89555** | [−0.89570, −0.89539] | **1.0** | **不支持，方向相反** |
| H6：Layer 2 entropy − Layer 8 entropy | +0.45994 | [0.45831, 0.46152] | 2.73e-253 | 支持 |
| H7：L01H12 broad-query fraction − 0.50 | +0.38248 | [0.37722, 0.38764] | 2.73e-253 | 支持 |
| H8：無 leading EOT 的 L08H03 first-content mass − 0.50 | +0.42384 | [0.41895, 0.42845] | 1.44e-250 | 支持 |

完整數字：[validation_tests.csv](validation_tests.csv)；所有 head：[head_statistics.csv](head_statistics.csv)；各類別及 discovery/validation 分開統計：[category_enrichment.csv](category_enrichment.csv)。

## 圖表與閱讀順序

所有下列圖都有同名 PDF；原 notebook 的 BERT 圖／程式未修改。

1. [Figure 1：五種 head 的真實 attention 連線](figures/figure1_head_examples.png)；固定 sample 0，不以最好看的驗證樣本挑圖。視窗外 mass 未重新分配，並另外提供 [完整矩陣](figures/full_head_examples.png)。
2. [Figure 2：token categories、boundary queries、relative positions](figures/figure2_token_categories.png)。
3. [驗證組：標點／邊界相對於 matched baseline](figures/validation_target_controls.png)，error bars 為文章 bootstrap 95% CI，三個 panel 的 y 軸不同。
4. [首位置集中、前一 token、排除 initial key 後的 entropy](figures/concentration_diagnostics.png)。
5. [Figure 4：raw entropy 與各自 causal-uniform reference](figures/figure4_raw_entropy.png)。
6. [Normalized entropy 與最大 attention](figures/entropy_and_max_weight.png)。
7. [邊界可見範圍與 query-position 分層](figures/boundary_and_position.png)，長位置 bins 標示实际文章數，避免讓長／短 contexts 不同組成被忽略。

Entropy 資料另外輸出為 [entropy_by_head.csv](entropy_by_head.csv) 與 [entropy_by_position.npz](entropy_by_position.npz)。

## 限制／可寫入報告的結論

**可主張：** 在這批 Wikipedia 相鄰雙段落中，GPT-2 存在可重現的標點／邊界 enrichment、首位置 attention 集中、近乎純 previous-token heads，以及 broad heads。低 entropy 並不能區分這些不同型態。

**不可主張：** 找到已證明的 functional no-op token；首位置效應與 token identity 已被因果分離；和 BERT 數值差異全由 causal masking 造成；或者所有 GPT-2 heads／資料分布都有相同行為。

- 只有單一 GPT-2 checkpoint、單一資料 snapshot／seed，且是英文 Wikipedia。Streaming shuffle 使用 buffer，不是對整個 Wikipedia 做精確的 simple random sampling。
- 842 篇是相對於先前 158 篇探索資料的本次保留組，並非跨語料、跨模型的獨立 replication。未做 optional stopping 或在驗證結果上重新挑 head。
- Double-newline category 只涵蓋此輸入格式的自然邊界，不是 BERT 的特殊 `[SEP]`；EOT 包住 sample 也不是完整重建 GPT-2 的 pretraining packing。
- 大多數 samples 遠短於 1024；513–1023 的位置區間僅有 21 篇，不能把短上下文結果包裝成長上下文結論。
- Distance matching 使用粗 buckets，且 expected mass 由同一 head/query 的觀測 distance profile 推得，是描述性對照，不是獨立生成的 causal null。
- 圖上的 full-corpus N=1000 為描述性；正式顯著性檢驗只使用固定 validation N=842。極小 p-value 不代表 effect size 大，更不代表功能已確認。
