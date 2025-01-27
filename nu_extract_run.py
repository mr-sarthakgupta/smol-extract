from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument


fetcher = LinkContentFetcher()

streams = fetcher.run(urls=["https://example.com/"])["streams"]

converter = HTMLToDocument()
docs = converter.run(sources=streams)

print(docs)

from haystack.components.generators import HuggingFaceLocalGenerator
import torch

generator = HuggingFaceLocalGenerator(model="numind/NuExtract-tiny",
                                      huggingface_pipeline_kwargs={"model_kwargs": {"torch_dtype":torch.bfloat16}})

# effectively load the model (warm_up is automatically invoked when the generator is part of a Pipeline)
generator.warm_up()

prompt="""<|input|>\n### Template:
{
    "Car": {
        "Name": "",
        "Manufacturer": "",
        "Designers": [],
        "Number of units produced": "",
    }
}
### Text:
The Fiat Panda is a city car manufactured and marketed by Fiat since 1980, currently in its third generation. The first generation Panda, introduced in 1980, was a two-box, three-door hatchback designed by Giorgetto Giugiaro and Aldo Mantovani of Italdesign and was manufactured through 2003 — receiving an all-wheel drive variant in 1983. SEAT of Spain marketed a variation of the first generation Panda under license to Fiat, initially as the Panda and subsequently as the Marbella (1986–1998).

The second-generation Panda, launched in 2003 as a 5-door hatchback, was designed by Giuliano Biasio of Bertone, and won the European Car of the Year in 2004. The third-generation Panda debuted at the Frankfurt Motor Show in September 2011, was designed at Fiat Centro Stilo under the direction of Roberto Giolito and remains in production in Italy at Pomigliano d'Arco.[1] The fourth-generation Panda is marketed as Grande Panda, to differentiate it with the third-generation that is sold alongside it. Developed under Stellantis, the Grande Panda is produced in Serbia.

In 40 years, Panda production has reached over 7.8 million,[2] of those, approximately 4.5 million were the first generation.[3] In early 2020, its 23-year production was counted as the twenty-ninth most long-lived single generation car in history by Autocar.[4] During its initial design phase, Italdesign referred to the car as il Zero. Fiat later proposed the name Rustica. Ultimately, the Panda was named after Empanda, the Roman goddess and patroness of travelers.
<|output|>
"""

result = generator.run(prompt=prompt)
print(result)

print('meow')
exit()


from haystack.components.builders import PromptBuilder
from haystack import Document

prompt_template = '''<|input|>
### Template:
{{ schema | tojson(indent=4) }}
{% for example in examples %}
### Example:
{{ example | tojson(indent=4) }}\n
{% endfor %}
### Text
{{documents[0].content}}
<|output|>
'''

prompt_builder = PromptBuilder(template=prompt_template)


example_document = Document(content="The Fiat Panda is a city car...")

example_schema = {
    "Car": {
        "Name": "",
        "Manufacturer": "",
        "Designers": [],
        "Number of units produced": "",
    }
}

prompt=prompt_builder.run(documents=[example_document], schema=example_schema)["prompt"]

print(prompt)


import json
from haystack.components.converters import OutputAdapter


adapter = OutputAdapter(template="""{{ replies[0]| replace("'",'"') | json_loads}}""",
                                         output_type=dict,
                                         custom_filters={"json_loads": json.loads})

print(adapter.run(**result))

from haystack import Pipeline

ie_pipe = Pipeline()
ie_pipe.add_component("fetcher", fetcher)
ie_pipe.add_component("converter", converter)
ie_pipe.add_component("prompt_builder", prompt_builder)
ie_pipe.add_component("generator", generator)
ie_pipe.add_component("adapter", adapter)

ie_pipe.connect("fetcher", "converter")
ie_pipe.connect("converter", "prompt_builder")
ie_pipe.connect("prompt_builder", "generator")
ie_pipe.connect("generator", "adapter")


urls = ["https://techcrunch.com/2023/04/27/pinecone-drops-100m-investment-on-750m-valuation-as-vector-database-demand-grows/",
        "https://techcrunch.com/2023/04/27/replit-funding-100m-generative-ai/",
        "https://www.cnbc.com/2024/06/12/mistral-ai-raises-645-million-at-a-6-billion-valuation.html",
        "https://techcrunch.com/2024/01/23/qdrant-open-source-vector-database/",
        "https://www.intelcapital.com/anyscale-secures-100m-series-c-at-1b-valuation-to-radically-simplify-scaling-and-productionizing-ai-applications/",
        "https://techcrunch.com/2023/04/28/openai-funding-valuation-chatgpt/",
        "https://techcrunch.com/2024/03/27/amazon-doubles-down-on-anthropic-completing-its-planned-4b-investment/",
        "https://techcrunch.com/2024/01/22/voice-cloning-startup-elevenlabs-lands-80m-achieves-unicorn-status/",
        "https://techcrunch.com/2023/08/24/hugging-face-raises-235m-from-investors-including-salesforce-and-nvidia",
        "https://www.prnewswire.com/news-releases/ai21-completes-208-million-oversubscribed-series-c-round-301994393.html",
        "https://techcrunch.com/2023/03/15/adept-a-startup-training-ai-to-use-existing-software-and-apis-raises-350m/",
        "https://www.cnbc.com/2023/03/23/characterai-valued-at-1-billion-after-150-million-round-from-a16z.html"]


schema={
    "Funding": {
        "New funding": "",
        "Investors": [],
    },
     "Company": {
        "Name": "",
        "Activity": "",
        "Country": "",
        "Total valuation": "",
        "Total funding": ""
    }
}

from tqdm import tqdm

extracted_data=[]

for url in tqdm(urls):
    result = ie_pipe.run({"fetcher":{"urls":[url]},
                          "prompt_builder": {"schema":schema}})

    extracted_data.append(result["adapter"]["output"])

print(extracted_data[:2])



def flatten_dict(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key} - {k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        elif isinstance(v, list):
            items.append((new_key, ', '.join(v)))
        else:
            items.append((new_key, v))
    return dict(items)

import pandas as pd

df = pd.DataFrame([flatten_dict(el) for el in extracted_data])
df = df.sort_values(by='Company - Name')

print(df)

import networkx as nx

# Create a new graph
G = nx.Graph()

# Add nodes and edges
for el in extracted_data:
    company_name = el["Company"]["Name"]
    G.add_node(company_name, label=company_name, title="Company")

    investors = el["Funding"]["Investors"]
    for investor in investors:
        if not G.has_node(investor):
            G.add_node(investor, label=investor, title="Investor", color="red")
        G.add_edge(company_name, investor)


from pyvis.network import Network
from IPython.display import display, HTML


net = Network(notebook=True, cdn_resources='in_line')
net.from_nx(G)

net.show('simple_graph.html')
display(HTML('simple_graph.html'))

