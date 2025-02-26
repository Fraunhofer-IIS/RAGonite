**Note:** The ConfQuestions benchmark and pipeline code will be released after internal OSS clearance, by 10.03.2025 (before the start of the conference).

# Publication artifacts

- Code: [src/](src/)
- ConfQuestions: [data/](data/)
- Poster: [artifacts/poster-wsdm.pdf](artifacts/poster-wsdm.pdf)
- Slides: [artifacts/poster-wsdm.pdf](artifacts/slides-wsdm.pdf)
- Demo video: Coming soon!
- Screenshots: Coming soon!
- Manuscript: https://arxiv.org/abs/2412.10571v3

If you have any questions, please send an email to:

* rishiraj DOT saha DOT roy AT iis DOT fraunhofer DOT de OR
* joel DOT schlotthauer AT iis DOT fraunhofer DOT de OR
* chris DOT hinze AT iis DOT fraunhofer DOT de

# Project overview

Retrieval Augmented Generation (RAG) works as a backbone for interacting with an enterprise's own data via Conversational Question Answering (ConvQA). In a RAG system, a retriever fetches passages from a collection in response to a question, which are then included in the prompt of a large language model (LLM) for generating a natural language (NL) answer. However, several RAG systems today suffer from two shortcomings: (i) retrieved passages usually contain their raw text and lack appropriate document context, negatively impacting both retrieval and answering quality; and (ii) attribution strategies that explain answer generation typically rely only on similarity between the answer and the retrieved passages, thereby only generating plausible but not causal explanations. In this work, we demonstrate RAGONITE, a RAG system that remedies the above concerns by: (i) contextualizing evidence with source metadata and surrounding text; and (ii) computing counterfactual attribution, a causal explanation approach where the contribution of an evidence to an answer is determined by the similarity of the original response to the answer obtained by removing that evidence. To evaluate our proposals, we release a new benchmark ConfQuestions: it has 300 hand-created conversational questions, each in English and German, coupled with ground truth URLs, completed questions, and answers from 215 public Confluence pages. These documents are typical of enterprise wiki spaces with heterogeneous elements. Experiments with RAGONITE on ConfQuestions show the viability of our ideas: contextualization improves RAG performance, and counterfactual explanations outperform standard attribution. An example of a canonical conversation for this work is provided below:

~~~
Question 1: What was the BIOS and Build versions used for Dell Optiplex 7040 in the OpenXT 9.0 measurement tests?

Answer 1: Build version: 6662 and BIOS version: 1.14.0

Question 2: And what about TPM?

Answer 2: Version 2.0

Question 3: Which OpenXT 9.0.2 tests did Dell 9010 pass for the legacy system?

Answer 3: OpenXT 8.0.2-pre ($1909), OTA to OpenXT 9.0.2 ($6768), and OpenXT 9.0.2-pre ($6768)

Question 4:	Which machines were tested with both legacy and EFI for OXT 9.0.2?

Answer 4: Dell Latitude 7470 and Dell Optiplex 7050

Question 5: Coming back to OXT 9.0, did the Optiplex 7060 pass the UEFI OTA upgrade 8.0.1 to 9.0.0 test?

Answer 5: No
~~~

**Motivation** "Talk to your data" is a major research theme today, where users interact with local knowledge repositories to satisfy their information needs. Conversational question answering (ConvQA) is a natural choice for such interactions, where a user starts with a self-sufficient (intent-explicit) question, and follows that up with more ad hoc, conversational questions that leave parts of the context unspecified (intent-implicit). Hand in hand, the emergence of powerful large language models (LLMs) has led to retrieval augmented generation (RAG) as the backbone for designing QA systems over one's own data. In RAG pipelines, given a question, a retriever fetches relevant evidence from local data, and this is passed on to an LLM for a concise and fluent answer to the user's question. Enterprise knowledge repositories usually consist of documents with heterogeneous elements, i.e. they often contain interleaved tables, lists and passages (henceforth, we use the unifying term evidence). Typical examples of such documents with mixed structured and unstructured elements are meeting notes, test reports, or product descriptions. The setting for this demo is ConvQA with RAG over such heterogeneous document collections.

**Limitations of state-of-the-art** Over the last four years, RAG has been a topic of intense investigation. Beyond literature, organizations like LangChain, LlamaIndex, or Cohere offer frameworks to build RAG systems. We posit that despite several advanced features, these suffer from two basic concerns, one at each end of the pipeline: (i) at the beginning, documents are typically split into chunks (usually one or more passages) that are indexed on their content. When these chunks are retrieved and fed to an LLM, they often lack supporting context from the document, which adversely affects both retrieval, and subsequent answering; and (ii) at the end, attribution mechanisms that attach provenance likelihoods of the answer to the retrieved units of evidence, are solely based on statistical similarity between the answer and the evidence: these are not causal, but rather only plausible explanations. Moreover, current pipelines support raw text and rarely mention how tabular elements could be handled.

**Contributions** We make the following salient contributions through this work:
- We demonstrate RAGONITE, a new RAG system that concatenates page titles, headings, and surrounding text to raw contents of evidences for better retrieval and answering;
- We compute counterfactual attribution distributions over retrieved evidences as causal explanations for answers;
- We bring tables under RAGonite’s scope by linearizing each record (row) via verbalization and similar techniques;
- We create ConfQuestions, a benchmark with 300 conversational questions for evaluating RAG-based heterogeneous QA. The benchmark and all other artifacts for this work are public at https://github.com/Fraunhofer-IIS/RAGonite.

## System outline

**Backend** An overview of the RAGONITE pipeline is in Figure 1. RAGONITE's backend is split into a functional core, which handles retrieval and answer generation, and a stateful layer that persists chats into a SQLite database and provides a REST API to the frontend using FastAPI. The dependencies in the functional core include the vector database (ChromaDB), and prompt template (Jinja) and LLM libraries (gpt-4o and Llama-3.1-8B). We use ChromaDB as our vector database for storing the contextualized evidences, and also use ChromaDB's in-built retrieval functions. The multilingual [BGE embedding model](https://huggingface.co/BAAI/bge-m3) was used to embed evidences, that was found to work slightly better than [text-embedding-3-small](https://openai.com/index/new-embedding-models-and-api-updates/) from OpenAI. While ChromaDB was used for its extensibility, we also explored variants like Weaviate and Milvus. We used top-k hybrid search with the [BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) and reciprocal rank fusion (RRF) being used to merge k dense retrieval and lexical retrieval results. We use GPT-4o as our LLM of choice inside the question completion and answer generation modules for efficiency and quality, but we also support a Llama model in our demo. All prompts can be seen inside the demo interface for transparency. A single GPU server (4x48GB NVIDIA Ada 6000 RTX, 512 GB RAM, 64 virtual cores) was used for all our experiments. All code is in Python.

![Figure illustrating RAGONITE on Confluence data](images/overview.png)
*Figure 1: The RAGonite workflow enhances RAG pipelines at both ends, preprocessing evidence and explaining answers.*

**Frontend** For the frontend we developed a single-page React application (actually Preact, as a more lightweight React-compatible alternative), intentionally avoiding additional dependencies to eliminate the need for a build process. All API calls in the RAGONITE demonstration are handled by the frontend.

## Demo walkthrough

![Figure showing RAGONITE screenshot](images/screenshot.png)
*Figure 2: An annotated walkthrough the of RAGONITE demo. Blue boxes guide the reader and are not part of the UI.*

We use a screenshot of the RAGONITE main page in Figure 2, on which we overlay numbered blue boxes, for a demo walkthrough. First, a user must select a domain (0) on which to use RAGONITE (our focus here is on enterprise wikis exemplified by Confluence, but RAGONITE also runs on other domains like soccer, automobiles, movies, and fictional universes). Then the user uses the question input box (1) to begin a conversation. Suppose we are at turn two (2): when RAGONITE receives a conversational question as input (like `And what about TPM?` in Turn 2 after `What was the BIOS and Build versions used for Dell Optiplex 7040 in the OpenXT 9.0 measuerment tests?` in Turn 1), it uses its question completion module to rephrase the question into an intent-explicit form using an LLM (3). While the question in the first conversation turn is usually self-contained, follow-up questions are completed using relevant information from previous questions and generated answers. Offline, the heterogeneous document collection is preprocessed via our evidence contextualization module into a pool of evidence where necessary document context is concatenated to the raw contents of the evidence. A retriever then takes the intent-explicit question and searches over the evidence pool, to return top-k question-relevant evidences (4, 4a-c) using the hybrid retrieval score (5). These top-k evidences are inserted into the prompt of another LLM instance to generate the answer (6). Finally, the generated answer and the top-k evidences are fed into our counterfactual attribution module. When a user clicks on "Explain" (6a), a command line window pops up to output the attribution distribution as an explanation of how the answer was potentially constructed. We log user feedback on the answer for future use (7). A user can see a trace through the pipeline using behind-the-scenes buttons (8): for example, retrieval results for lexical and dense search, LLM prompts, and more. Users can adjust retriever and generator configurations (9), see their past conversations (10), including deleted ones (11). Follow-up question suggestions are also provided (12) to help domain exploration. Notably, most open-source RAG demos only consist of the evidence retriever and answer generator modules. On average, RAGONITE requires about one second to answer a question, and approximately two seconds for explaining the answer.

## Evidence contextualization

![Figure motivating need for evidence contextualization](images/webpage.png)
*Figure 3: Toy wiki page with heterogeneous elements to motivate evidence contextualization. A question like `todo for alice in oct rag meeting?` can only be faithfully answered by joining information in the relevant table row, table footer, page title, and preceding heading and text. This implies that unless the evidence as stored in the DB contains the supporting context, there is no chance that a retriever can fetch it from the corpus with the question as a search query.*

**Idea** Evidence contextualization in current RAG systems mainly involve steps like: (i) coreference resolution for resolving simple pronouns like (s)he, her, his, etc., and (ii) overlapping text chunking involving sliding windows over passages. These have some basic limitations like: (i) resolving pronouns is mainly limited to previous entity references, and is not enough for more subtle coreferences like "these configurations" or "the previous model"; (ii) sliding word/token windows do not generalize well to structured evidence like tables. We, however, adopt a simple but effective alternative of concatenating document context to each evidence at indexing time.

A toy example of an enterprise wiki page is in Figure 4. We begin by capturing text inside `<table> ... </table>` tags as tables. Text inside `<ol>...</ol>` (ordered list) and `<ul>...</ul>` (unordered list) tags are stored as lists. Each span of the remaining text between any `<heading>`-s, or between a `<heading>` and the beginning or end of the document, is assigned to one passage. Each list and each paragraph become individual pieces of evidence.

**Preprocessing** We store each table in a verbalized form, which converts structured evidence to a form more amenable to an LLM prompt while retaining scrutability by a human. In this mode, we traverse left to right in a table row, and linearize the content as :"<Column header 1> is <value 1>, and <Column header 2> is <value 2>, ...". We prepend this text with "Row <id> in Table <id>" to preserve original row ordering information as well as table ordering relative to the page where the content is located. For instance, the second row in the table in Figure 4 would be verbalized as "Row 2 in Table 1: Member is Alice, and Task is Similarity function, and ...". Along with the complete table, each linearized row is also stored as an individual evidence. This helps pinpoint answer provenance to individual rows in tables when possible, as well as resolve comparative questions that are easier when the whole table is in one piece in the prompt.

**Contextualization** At this point, we have passages, lists, tables, and table records as individual evidences. We then concatenate the following items to each piece of evidence: page title, previous heading, the evidence before, and the evidence after. Both the raw textual forms and their embeddings are indexed in our database, enabling lexical and dense search, respectively. Note that indexing whole pages and inserting full text of top-k documents into an LLM prompt cannot easily bypass evidence contextualization, due to arbitrary lengths of web documents: operating at evidence-level is therefore a practical choice.

![Figure showing evidence contextualization in demo](images/magic.png)
*Figure 4: Contextualized evidence as in answer prompt.*

An example of contextualized evidence is in Figure 4 (inside dashed box): the raw content of the evidence (the seventh row in the first table) starts with "Row 7 in Table 1 ..." But the previous content starting at "OpenXT 9.0 ..." consists of the page title, preceding heading, and preceding text. "OTA upgrade" at the end comes from adding succeeding text (the table footer here). An idea sketch for the contextualization module is in Figure 5.

![Figure showing idea sketch of evidence contextualization](images/idea-sketch-1.png)
*Figure 5: Idea sketch for evidence contextualization.*

## Counterfactual attribution

**Idea** Counterfactual explanations through passage perturbation has been explored in the context of RAG and more generally for LLM understanding, but not for answer attribution and not via evidence removal, as explored in this work. The intuition here is that a definitive way of saying how much an evidence contributed to an answer is to remove the evidence from the prompt, and see how much the original answer 𝑎changed. If it did not change much, then the evidence is unlikely to have played any major role, and vice versa.

![Figure showing CFA algo](images/attrib-algo.png)
*Figure 6: A formal representation of the counterfactual attribution algorithm.*

**Counterfactual Attribution (CFA)** This simple idea is the core of our proposed algorithm (formal representation in Figure 6). The only other confounding factor is the presence of redundant evidences in the prompt: the retriever may sometimes fetch two (or more) semantically equivalent evidences if they both satisfy the user's intent. In the context of counterfactuals, it may be that when one of these redundant evidences are removed, the answer hardly changes. This may lead to the wrong conclusion that the first evidence was unimportant: it just happens that the second evidence makes up for it content-wise. Hence, it is important that we first cluster evidences by content before we apply an iterative removal of each evidence cluster. Each cluster's removal and corresponding counterfactual answer generation is implemented in parallel inside the demo for efficiency, as evidences (clusters) can be processed independent of each other. Moreover, to adjust for non-determinism in LLM generations, we use a Monte Carlo method: in this case it simply means that we repeat the answer generation a few times, compute similarities between the answer and the counterfactual answer in each iteration, and average these values for each cluster. One minus this average similarity is the contribution of each evidence cluster to the original answer: the more similar the answers are, the less an evidence contributed.

These raw answer contribution scores for each evidence cluster are then normalized via masked softmax to derive the final attribution distribution (example in Figure 7). We used the popular DBSCAN algorithm for clustering evidences as it requires only two tunable parameters, and does not require the specification of the number of clusters upfront (hard to predict in our case). Cosine similarity between the text embeddings of the answer and the counterfactual answer (in both cases the completed question from the benchmark is prepended for providing the context in which the similarity is sought) was used as our similarity function. Using [JinaAI embeddings](https://huggingface.co/jinaai/jina-embeddings-v3) to encode evidences which was found to work better than other embedding models (BGE, OpenAI) in this context. While our algorithm produces fairly accurate results, output distributions may sometimes appear to be somewhat degenerate to end users. For example, say among six clusters, the top-evidence cluster is attributed 20% and the remaining five groups are all assigned about 16%. We thus introduce a temperature parameter to introduce a higher visible skew of the distribution towards the top-scoring evidence cluster. Answers and counterfactual answers were generated using the cheaper GPT-4o-mini model (instead of GPT-4o) due to budget constraints: every question entails the generation of up to ten counterfactual answers, which entails a substantially higher number of output tokens than generating the intent-explicit questions or the original answers (where we use GPT-4o).

![Figure showing CFA output](images/attrib-output.png)
*Figure 7: Answer explanation by counterfactual attribution.*

An idea sketch for the attribution module is in Figure 8.

![Figure showing idea sketch of counterfactual attribution](images/idea-sketch-2.png)
*Figure 8: Idea sketch for counterfactual attribution.*

## The ConfQuestions benchmark

**Need for a new benchmark** While there are many public QA benchmarks, there were none that fit all our desiderata: one that uses heterogeneous pages in enterprise wikis, is suitable for ConvQA with RAG, and contains complex questions in more than one language. So we created our own benchmark ConfQuestions as follows. We crawled 10 Spaces under https://openxt.atlassian.net/wiki/spaces, yielding 215 public Confluence pages from the developer Atlassian. These contain many documents that are very close to each other content-wise (like test reports for software versions 6−9, organizational policy over several years, and several meeting notes on a topic), yet each page contains unique information: this is very suitable for evaluating accurate retrieval and generation models. The pages contain a mix of concepts like software compatibility, build configurations, planning, along with mentions of people, dates and software, manifested via heterogeneous elements like tables, lists, and passages. These documents thus nicely simulate an enterprise Wiki space, our focus here.

**Benchmark construction** The authors went through every page to judge the feasibility of generating NL conversations over them. Eventually, 50 conversations were generated: 80% with 5 turns, and 20% longer ones with
10 turns (making context handling tougher), so 300 questions in all. 20% conversations also span two URLs instead of one, containing slight topic shifts. The annotators provided questions, intent-explicit versions, answer source (passage/list/table), complexity type (simple/complex), and ground truth answers along with source URLs. The questions were originally generated in English (EN). Native German speakers then translated both conversational and completed questions by hand to German (DE). We took special care to make the benchmark challenging from two aspects: (i) 50% of the questions are complex in the sense that they either require joining multiple evidences or involve aggregations, negations, or comparisons; and (ii) the answers to these questions are spread equally across passages, lists, and tables. So to be able to perform well overall on ConfQuestions, a RAG systems needs to be able to handle complex questions over heterogeneous elements in multiple languages. Benchmark statistics are in Table 1.

![Table showing ConfQuestions statistics](images/confquestions-stats.png)
*Table 1: Statistics of the ConfQuestions benchmark.*

## Results and insights

### Metrics

We use Precision@1 as our retrieval metric (since there is usually one gold URL), and Answer relevance as judged by GPT-4o (0 for non-relevant, 0.5 for partially relevant, 1 for relevant) against gold answers, as our metric for answer quality. Question-wise results are averaged over the full dataset. We use accuracy for evaluating attribution explanations as follows: we find the evidence that has the highest attribution score (for a cluster, the score of the cluster is considered to be the score of each evidence inside), and find its source URL. If this URL matches that of the gold answer, then accuracy is
1, else 0. The average accuracy over all questions is then reported. This is interpretable as a percentage if multiplied by 100.

For both Precision@1 and Attribution accuracy, we match the URL of the top-evidence against the answer URL in the benchmark. This is a proxy for ground truth: a perfect case would be to have gold labels at evidence-level, but this is very difficult to annotate for our benchmark, as gold answers are often composed of nuggets of information spread over the entire document.

### Setup

We now run and evaluate RAGONITE in various configurations over the 600 English and German questions in ConfQuestions. The proposed default configuration uses: (i) all available context; (ii) all evidences as the corpus; (iii) LLM-generated intent-explicit questions; (iv) verbalization as the linearization technique for table records; (v) both row- and table-level indexing for table elements; (vi) GPT-4o as our LLM of choice; (vii) BGE as our text embedding model; (viii) hybrid search as our retrieval mode; and (ix) BGE+RRF reranking over the hybrid search results. We feed the reranked top-10 evidences (k=10) by the retriever into the LLM.

**Prompts** There are three prompts used in RAGonite: (i) for rephrasing a conversational question; (ii) for generating an answer from retrieved evidences; and (iii) for evaluating a generated answer. These are presented in Figures 9, 10, and 11, respectively (used for both English and German questions).

![Figure showing rephrasing prompt](images/prompt-rephrase.png)
*Figure 9: LLM prompt for rephrasing a conversational question into an intent-explicit form.*

![Figure showing answering prompt](images/prompt-answer.png)
*Figure 10: LLM prompt for generating an answer from the retrieved pool of evidence.*

![Figure showing evaluating prompt](images/prompt-evaluate.png)
*Figure 11: LLM prompt for evaluating a generated answer according to the Answer Quality measure.*

### Key findings

![Table showing contextualization results](images/results-1.png)
*Table 2: Contextualization configurations of RAGONITE. The highest value in each row in a group is in bold.*

**Evidence contextualization improves performance** Contextualization results are in Table 2. We make the following observations: (i) Adding all the suggested contexts (ALL) consistently led to significantly improved performances for retrieval and generation. This means that enhancing evidences from documents with surrounding context is generally beneficial for RAG over collaborative wikis: content authors are often efficiency-oriented and do not repeat information inside passages, lists, or tables that are already mentioned in page titles or previous headings, for example. Similarly, adding preceding and succeeding texts, give coherence to the evidence and help in better retrieval (users often use localizing words in questions that may not lie exactly inside the pertinent evidence but could be nearby in the document). Surrounding texts clearly also help the LLM generate more consistent and relevant responses; (ii) Systematically, the page title (TTL) was found to be the single most helpful context to be added to the evidences (0.483 P@1 for TTL vs. 0.528 for ALL, and 0.477 Answer Relevance for TTL vs. 0.529 for ALL), with each of the other three individual add-ons bringing in substantial improvements over NONE (ALL > TTL > AFT > BEF > HDR > NONE for P@1 and ALL > TTL > BEF > HDR > AFT > NONE for answer relevance). No feature hurts performance. It is notable that succeeding content also helps (a notable case is that of table footers), while common coreferencing methods only contextualize using preceding text; (iii) These improvements are systematic with respect to question complexity (simple/complex), answer source (passage/list/table), and language (English/German), demonstrating that contextualization is a worthwhile operation across diverse question types. We also note that the biggest gains in retrieval (0.220 gain in P@1 of ALL over NONE) and answering (0.179 gain in answer relevance of ALL over NONE) performance are achieved for answer-in-list questions – a very common QA scenario for enterprise wikis where one asks for meeting notes, attendees and action items – all often stored as lists.

![Table showing attribution results](images/results-2.png)
*Table 3: Accuracies of explanation generation methods.*

**Counterfactual attribution is effective** While counterfactual evidence has been proven to be a tool for discovering causal factors, our particular implementation still needed proof of efficacy. Results in Table 3 (Column 4) show that we consistently reach accuracies close to 80%, definitely an acceptable number in terms of user trust in RAGONITE explanations. The numbers in parentheses in Column 1 show over how many questions the respective averages were computed in Columns 2 − 4. For questions where the gold document/evidence was not retrieved in the top-10, there is no chance that the attribution module could spot the correct evidence. Hence these are removed from this evaluation and we do not see round numbers like 600/200/100 as in Table 2. Looking at individual data slices across rows, it is satisfying to observe that the high performance was not due to skewed success on easier slices: improvements over standard or naive attribution are quite systematic over almost all criteria (varying only within a narrow span of 77−82%). In the naive mode, we compute cosine similarities between the answer and the evidences, and then use softmax scores over this distribution as the attribution scores for the corresponding evidences. The top scoring evidence is used to compute accuracy. Within the counterfactual approaches, the one with clustering generally does better (with-clusters best in 5/8 cases, without-clusters in 2/8: marked in boldface in Table 3). Understandably, complex questions and answer-in-table questions are slightly more difficult cases to solve (relatively lower accuracies of about 77−78%). Performance for German was found to be slightly better than that for English.

### In-depth RAG analysis

![Table showing RAGONITE ablations](images/results-3.png)
*Table 4: RAGonite ablation study. Highest values in rows in bold.*

In Table 4, we report several drill-down experiments with RAGONITE configurations. Whenever one configuration choice was altered, the remaining values were held constant as per our default configuration. The study leading to this table was also used to *select* our default configuration. Specific observations are listed below:

* **[Conversation turns]** In Rows 1−3, we observe that while there is an understandable drop after the first turn owing to intent-implicit questions, RAGonite remains fairly consistent in performance in deeper turns (default configuration used).
* **[Answer source]** In Rows 4−6, we see clear proof that heterogeneous evidence helps QA: the ability of tapping into a mixture of sources notably improves RAG performance.
* **[Question completion]** In Rows 7−9, we note that accurate question completion is still a bottleneck, as using human completions from the benchmark substantially improves metrics.
* **[Table linearization]** In Rows 10−12, verbalization of table records indeed makes table contents more retrievable by a hybrid retriever (see below) as well as more digestible by an LLM, compared to natural linearization alternatives. But we also prefer verbalizations in the RAGonite interface owing to its scrutability by an end-user.
* **[Table indexing]** In Rows 13−15, we find that retaining both table embeddings and individual row embeddings in the DB is preferable to having only either of these. This is understandable in the sense that some questions need an aggregation over several rows or the entire table, this is difficult if the full table is not indexed (and retrieved). On the other hand, questions pertaining to specific cells or joins concerning 1−2 rows may be harder to fish out from very large tables by an LLM: here the ability to retrieve individual rows is helpful.
* **[Generation LLM]** In Rows 16−18, we find that using GPT-4o led to superior answer quality than Llama3.1-8B. Unfortunately the presence of large lists and table evidences in Confluence often led to the context window being exceeded for Llama. The same LLM was used for both completed question and final answer generation. The P@1 is also lower for Llama (0.480 vs. 0.528 for GPT-4o) because the retrieval is carried out via the intent-explicit question generated by Llama (slightly inferior to corresponding completions from GPT-4o). As requested in the prompt, the out-of-scope (OOS) message "The desired information cannot be found in the retrieved pool of evidence." was triggered  199/600 times (33.2%) for the default RAGonite configuration with GPT-4o. Computing the Precision@10 to be 60.7%, an OOS message would be expected  39.3% of the times. Since this is very close to the actual measured rate of the OOS message, we can say that GPT-4o showed high instruction-following capabilities and did not use its parametric knowledge.
* **[Embedding model]** In Rows 19−21, we see that using bge-m3 embeddings for encoding text led to marginally better retrieval (P@1=0.528) than the recent text-embedding-3-small model from OpenAI (P@1=0.525). While using the OpenAI model led to slightly better answer quality, better retrieval is the primary purpose of embeddings, leading to our choice of BGE as default. Moreover, BGE is used via local deployment, that we find preferable to the OpenaI API due to latency and cost issues.
* **[Evidence ranking]** In Rows 22−24, hybrid search was found to be superior than lexical or dense search built inside ChromaDB. We used 10 evidences each from lexical and dense search.
* **[Evidence reranking]** In Rows 25−27, reranking evidences in the common pool of lexical and dense retrieval via the multilingual BGE reranking model+RRF to create the hybrid ranked list, was found to be better than using only RRF. In RRF, the final list is created by ordering via a new evidence score that is the sum of the reciprocals of the ranks of the evidence in the lexical and dense rankings (a parameter k=60 is added to the original rank before computing the reciprocal).

## Summary

In this article, we show that RAGonite is an explainable RAG pipeline, which makes every intermediate step open to scrutiny by an end user. We propose flexible strategies for adding context to heterogeneous evidence, and causally explain LLM answers using counterfactual attribution. Future work will introduce cost and time-efficient agentic workflows.
