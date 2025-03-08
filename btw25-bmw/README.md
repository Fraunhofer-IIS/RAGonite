# Publication artifacts

- Code: [src/](src/)
- Data: [data/](data/)
- Poster: [artifacts/poster-btw.pdf](artifacts/poster-btw.pdf)
- Slides: [artifacts/slides-btw.pdf](artifacts/slides-btw.pdf)
- Demo video: https://vimeo.com/1063623916/fe9e6378c2?share=copy
- Screenshots: [artifacts/screenshots](artifacts/screenshots/)
- Manuscript: https://arxiv.org/abs/2412.17690

If you have any questions, please send an email to:

* rishiraj DOT saha DOT roy AT iis DOT fraunhofer DOT de OR
* chris DOT hinze AT iis DOT fraunhofer DOT de OR
* joel DOT schlotthauer AT iis DOT fraunhofer DOT de

# Project overview

Conversational question answering (ConvQA) is a convenient means of searching over RDF knowledge graphs (KGs), where a prevalent approach is to translate natural language questions to SPARQL queries. However, SPARQL has certain shortcomings: (i) it is brittle for complex intents and conversational questions, and (ii) it is not suitable for more abstract needs. Instead, we propose a novel two-pronged system where we fuse: (i) SQL-query results over a database automatically derived from the KG, and (ii) text-search results over verbalizations of KG facts. Our pipeline supports iterative retrieval: when the results of any branch are found to be unsatisfactory, the system can automatically opt for further rounds. We put everything together in a retrieval augmented generation (RAG) setup, where an LLM generates a coherent response from accumulated search results. We demonstrate the superiority of our proposed system over several baselines on a knowledge graph of BMW automobiles. An example of a canonical conversation for this work is provided below:

~~~
Question 1: What is the Coupe series all about? (Abstract intent)

Answer 1: The BMW Coupe series is all about performance, luxury, and sporty design, offering a more ...

Question 2: What fuel types does it support? (Lookup intent)

Answer 2: BMW Coupes support multiple fuel types, depending on the variant. Here’s a breakdown ...

Question 3: Seems like sleek sedans. Do they have only rear wheel drive? (Lookup intent)

Answer 3: No, BMW Coupes are not limited to rear-wheel drive. They also offer all-wheel drive ...

Question 4:	Can you tell me a bit about the transmission systems in Coupe models? (Lookup intent)

Answer 4: BMW Coupe models come with a mix of automatic and manual transmissions ...

Question 5: The Coupe models are sleek. Are these taller than an average X2 model? (Complex intent)

Answer 5: No, BMW Coupe models are generally lower in height compared to the BMW X2 ...
~~~

**Querying knowledge graphs** It is common practice in large-scale enterprises to organize important factual data as RDF knowledge graphs (KGs, equivalently knowledge bases or KBs). Storing data as RDF KGs has the advantage of a flexible subject-predicate-object (SPO) format, which simplifies the work of moderators by eliminating the need for complex schemas as in equivalent databases. KGs are usually queried via SPARQL, a data retrieval and manipulation language tailored to work with graph-based data. With the advent of large language models (LLMs), LLMs are now used to generate SPARQL queries from the user’s natural language (NL) questions, replacing previous sophisticated in-house systems (see Figure 1).

![Figure illustrating KG-QA with an LLM](images/traditional.png)
*Figure 1: Workflow in a typical KG-QA system with an LLM.*

**A two-pronged pipeline** Like other knowledge repositories, KGs are also conveniently searched and explored via conversational question answering (ConvQA) systems. However, ConvQA comes with a major challenge: vital parts of the question are left implicit by the user, as in dialogue with another human. In preliminary experiments, we found that even with the most capable LLMs like GPT-4o, translating a user’s conversational questions to SPARQL is a bottleneck, even with representative in-context examples. This problem is exacerbated for more complex intents with conditions, comparisons and aggregations. Moreover, a KG often contains information that can satisfy more abstract intents: this cannot be harnessed via SPARQL. In this work, we overcome these limitations and present a novel system with two branches: SQL over induced databases satisfy crisp information needs – including mathematical operations, while less defined intents are handled by text retrieval over KG verbalizations. Notably, we allow repeated retrievals if a single round fails to fetch satisfactory information from the backend KG. Finally, the results of these branches are merged by a generator LLM to formulate the answer that is shown to the user who posed the conversational question.

## RAGONITE: System Description and Unique Features

**System overview** Figure 2 shows an overview of our agentic system RAGONITE (Retrieval Augmented Generation ON ITErative retrieval results). There are two retrieval branches in RAGONITE: (i) the SQL query executes over the DB and obtains results, and (ii) the KG verbalizations are searched via a dense retriever model for passages that best satisfy the NL question. A plausible user question to our system could be `What is the average acceleration time to 100 kmph for BMW Sport models?`, followed by another question with implicit intent, such as `And how does this compare to a typical X1 model?` The system handles such queries by generating an intent-explicit SQL query and an intent-explicit NL question with an LLM. An intent-explicit form is a self-sufficient variant that contains all necessary information from the previous conversational context. The generated SQL query is then run over a database that we automatically derive from the KG, and the NL question is searched over the verbalizations of the KG facts. Comparing SQL results and the top-k verbalizations with the conversational question and the previous history, the LLM decides whether the retrieved information is sufficient to generate a satisfactory answer. If not, it suggests another round of retrieval. Finally, the accumulated SQL results and top-k verbalizations are fused by a second LLM agent to present a coherent and fluent answer to the end user. Our demo supports locally hosted LLMs as well as those accessible via APIs.

![Figure illustrating RAGONITE pipeline](images/overview.png)
*Figure 2: Workflow in RAGONITE, our proposed RAG pipeline.*

### Database induction

A key feature of this work is the derivation of a database from an RDF KG as follows.

**Inducing schema** KGs are often stored in a plain [RDF NTriples format](https://www.w3.org/TR/n-triples/). This means that it is stored as a flat set of facts that are SPO triples, where subjects are entities (like bmw-x6-m-competition, bmw-ix1-edrive20-sport, or adaptive-LED-headlights), predicates are relations (like engine-specification, wheelbase, or price), and objects are types (like car, engine and equipment) or literals (constants like 37450 EUR, 2760 mm or 250 kWh). We first convert the KG from Ntriples into [RDF Turtle format](https://www.w3.org/TR/turtle/) (see Figure 3), that encapsulates all facts of a specific subject entity (i.e. for grouping relevant facts that share the same subject, to induce some coherence into the flat set of facts). We identify unique entity types based on the values of the type predicate. For each entity type a table is added to the database. The subject entity is used as the primary key and each literal value is added as a column (e.g. "price" or "height" are added to the "car" table). Next, the relations between entity types are analyzed. For each 1:1 or 1:N relation between entity types A and B, a column is added to the entity table B that contains a foreign key which points to table A. For N:M relations, an additional table is added to the database which contains two columns with foreign keys.

![Figure illustrating RDF Turtle](images/kg-snippet.png)
*Figure 3: Knowledge graph snippet as Turtle RDF.*

**Inserting data** For each set of encapsulated Turtle facts, a single row is added to the respective table. The primary key is the subject entity in the Turtle facts. All literal values are added to fill the columns. Next we add the foreign keys of related entities. Then, for each table, the values of each column are analyzed to derive the data type of the column (like "INT" for "price", or "TEXT" for "drive type") and whether the values can be NULL. This information is then added to the schema.

**Enhancing semantics** Finally, our system allows renaming tables and columns to improve the LLM’s comprehension of the schema (such as "engine specification" maps to "engine"). Additionally, comments can be added to each column, for example, NL explanations of complex predicate names (like "WLTPCO2Emissioncombined").

To derive the intent-explicit SQL query from a conversational question, the LLM receives all previous questions and answers, the current question, and the database schema (all CREATE TABLE statements including data type constraints and comments). A toy example of this database induction process is in Figure 4. Note that if one were to manually create an equivalent DB from the KG, this could be result in simpler schema: our focus in this work is on automating the process, and finding the DB with least complexity is a topic for future work.

### KG verbalization

Abstract questions like `Innovative highlights in x7?` require NL understanding and common sense reasoning to map the intent to equipments and accessories of the BMW X7. We address such questions by verbalizing KGs, so that LLMs can reason over them like NL text. We again use the KG in Turtle format, where all facts with a specific entity as subject are grouped together. The facts in each Turtle "capsule" are then verbalized into NL passages using simple rules: (i) all schema-related prefixes are stripped to extract the raw content; (ii) the subject string, predicate string and object string are concatenated to form a sentence; (iii) "is" is added before type predicates, and "has" before other predicates; and (iv) The reverse formulation of the fact is also added to the passage to later help the generator LLM answer questions when the contained information is requested in complementary ways (`What is BMW 120 Sport's engine performance?` and `Which BMW engines have a performance of 125 kW?`). Explicitly adding reversed facts is helpful for smaller LLMs.

A toy RDF Turtle capsule looks like:
~~~
<engine/bmw-120-sport> a ns1:EngineSpecification ;
    ns1:enginePerformance "125 kW";
    ns1:fuelType <fuel-type/gasoline> .
~~~
This is verbalized into an NL passage as follows: "BMW 120 Sport is Engine Specification. BMW 120 Sport has engine performance 125 kW. 125 kW is engine performance of BMW 120 Sport. BMW 120 Sport has fuel type gasoline. Gasoline is fuel type of BMW 120 Sport." See also the bottom of Figure 4 (correspondig KG snippet at the top) for another toy example of verbalization, indicating how more abstract car features in the KG (consider `luxury features in saloon m?` as a question) come in scope of retrieval and answering, via NL linearization.

![Figure with toy example for DB induction](images/db-induction.png)
*Figure 4: Toy example showing DB induction and RDF verbalization.*

### Iterative retrieval

It is possible that results from either branch of RAGONITE are not of sufficient quality. For example, the SQL tool might generate a partial query or the text results may be incomplete or contain only irrelevant information. For such cases, the LLM can request additional rounds (set to three in our system, but configurable) of retrieval from either branch. Retrieval results, and possibly error messages from one round are included in the prompt in the next round. Sending such errors back to the retrievers gives them the opportunity to correct small mistakes, like resolving an ambiguous column name (for SQL), and guiding subsequent searches towards more relevant evidence (for text search). We further force the LLM to use both branches at least once to ensure as comprehensive a response as possible. This iterative workflow with tools is shown in Figure 5.

![Figure with tool use](images/tool.png)
*FIgure 5: Control flow with tool use in RAGONITE pipeline.*

### Branch integration

At the end of iterative retrieval, a second LLM agent generates an answer using both the SQL outputs and the top passages being inserted into its prompt. This way we do not have to decide which branch to prefer, neither at question-time nor after obtaining an answer from each mode: with both results at its disposal, the LLM can decide how to integrate them. The integration of branches and judgment over individual results is thus automated via the LLM and are not hard-coded into the system.

### Open LLM support

The default LLM in RAGONITE is GPT-4o with API access. A desideratum of agentic LLM systems is to ensure data security via local, on-premise LLM hosting. Along these lines, RAGONITE supports local deployment of open LLMs such as Llama-3.3. For our experiments, we hosted the 70B version with 4 bit-quantization using [ollama](https://ollama.com/library/llama3.3).

### Heterogeneous QA

Since one of our retrievers runs on NL verbalizations of the KG, it is straighforward to insert additional text contents (for example, Web documents with supplementary information to what is contained in our KG) to our backend knowledge repository. In one version of the demo, we incorporated about 400 English passages from the [BMW website](https://www.bmw.co.uk/en/index.html) to increase the scope of answerable questions: RAGONITE then becomes capable of heterogeneous question answering with a mix of a KG and a text collection as its knowledge sources.

## Demonstration overview

The complete code for our RAGONITE software cannot be released as per organizational policy. But we provide a short walkthrough with a typical example in this outline text. The UI in RAGONITE (see Figure 6) is divided vertically into four panels. Left to right, (i) the first panel stores previous chats; (ii) the main chat panel takes questions as input and displayes system answers; (iii) the third panel shows answer derivation steps; and (iv) the last strip helps switch between different RAGONITE configurations.

![Figure with RAGONITE screenshot](images/screenshot.png)
*Figure 6: Screenshot from the RAGONITE demonstrator showing a typical intermediate state within a conversation.*

**Walkthrough** We go through only the main workflow here: everything else can be explored by a user during an interactive demo session. Offline, the backend KG is preprocessed to derive the equivalent DB and the text corpus. At runtime, in Step 1, users enter their conversational question (potentially intent-implicit) in the input box provided. Suppose we are at the second conversation turn now (`Sounds like these are sleek sedans. Is an average X1 taller than a Coupe?`). In Step 2a, this question is passed on to the intent-explicit SQL-formulating LLM call. In this case, the LLM decides that results of one iteration (that only retrieved heights of BMW X1 variants) were not enough to satisfy the intent in this complex question, and so it formulates a second query (this time fetching the heights of Coupe models). At this moment, the iterator stops SQL querying and invokes the text retriever (Step 2b) via an intent-explicit NL question (`BMW X1 average height compared to BMW Coupe Models`). Note that Steps 2a and 2b can happen in an arbitrary order. A sample verbalization of a Turtle capsule can be seen in the screenshot below the corresponding blue box (for the entity bmw-x1-xdrive-25e-sport, we lowercase all text for simplicity), along with the retrieval score assigned by the vector search model (0.017). The iterator decides that contents of one round of text retrieval has enough information pertinent to the question, and by now both tools have been explored, so it hands over these results to the answer generator. This LLM agent crafts the final response shown to the user (Step 3). Through explicit prompting, we make the LLM inject source citations in its answer for scrutability. In this case, sources [1-2] refer to SQL results and [3-7] to verbalizations (understandable by labels above respective results). Prompts for all LLM calls can be examined by an user via the answer derivation panel.

**Backend** The core of RAGonite's backend handles query formulation, SQL execution, text retrieval, iteration/tool selection, and answer generation. The backend also has a stateful layer that stores conversations in an SQLite DB and provides a REST API to the frontend with FastAPI. Core dependencies include the vector database (ChromaDB), a template engine used for prompts (Jinja), and LLM libraries (OpenAI and ollama). ChromaDB is used as our vector DB for storing RDF verbalizations. We also use ChromaDB's in-built vector search. For all of our experiments, we ran RAGONITE on a shared GPU server (4x48GB NVIDIA Ada 6000 RTX, 512 GB RAM, 64 virtual cores). A single GPU was dedicated to running Llama-3.3-70B. If the LLM used for text inference is hosted in the cloud, for example, by using an OpenAI model, the remaining resource requirements are minimal and the RAGONITE instance can also run on a standard laptop.

**Frontend** We created a single-page React app, intentionally avoiding further dependencies to preclude the need for a build process. All API calls are handled by the frontend.


## Evaluation

**Setup** Our BMW KG (scraped from `bmw.co.uk`) has 3442 facts, 466 unique entities, 27 predicates, 7 types, and 1295 literals. So we get 7 tables and 466 passages via DB induction and RDF verbalization. The authors generated 6 conversations with 5 turns each, and test 4 GPT-4o-based configurations with them: (i) SPARQL-only, (ii) SQL-only, (iii) verbalization-only, and (iv) SQL+verbalizations. Questions were in 3 categories (10 in each): (i) lookup intents (`cost of 530em sport saloon?`), (ii) complex intents (`charging time of 225e Active Tourer less than avg over all bmws?`), and (iii) abstract intents (`luxury features in x5?`). Examples of these question types can also be seen in the canonical conversation towards the beginning of this article. Answers were marked correct/incorrect (SQL/SPARQL query must also be correct).

**Results** The two-pronged iterative approach did the best (28/30 questions correct). SQL-only got 18 (failure to handle abstract intents) and verbalization-only got 24 cases right (failure to handle complex math). A baseline where GPT-4o generated SPARQL queries, got only 4 answers correct. A particular vulnerability of SQL and SPARQL was incorrect entity linking for ad hoc mentions (like "gran coupe sport 220 i m" in questions, while the exact entity label is bmw-220i-m-sport-gran-coupe), and this is where verbalizations are particularly helpful. This also proved that LLMs are still much weaker at SPARQL generation than SQL (presumably due to lower training data), especially when it comes to complex or abstract intents in conversational question answering.

**Runtimes** We recorded stepwise runtimes for each of our 30 benchmark questions using our default pipeline with GPT-4o. Including question completion, the entire retrieval took 3.936 seconds, and answering took 2.409 seconds. Zooming in on the retrieval step, tool selection took 1.047 seconds, SQL execution took 0.095 seconds, and vector search took 0.539 seconds. The entire RAG pipeline thus took 6.348 seconds per question, on average. RAGONITE thus operates with interactive response times.

## Summary

In this article, we explain RAGONITE, a transparent RAG pipeline that adopts a novel alternative approach to ConvQA over KGs by iterative retrieval and merging results of structured and unstructured querying. While Text2SPARQL has been investigated in the past, we adopt a novel alternative of exploiting Text2SQL, where LLMs do much better, over a DB that is programmatically induced from the KG. By converting a KG to its verbalized form that is indexed as a vector DB, we show that the information in knowledge graphs can be useful for more abstract intents as well. RAGONITE contains an agentic workflow: key future work would enhance this further by incorporating reflection mechanisms that enable each module to critique and improve its output, without compromising efficiency. Another direction would be to fine-tune specific pipeline components for a RAG setting with synthetic data.
