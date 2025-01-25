from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.generators import HuggingFaceLocalGenerator
import torch
import json

fetcher = LinkContentFetcher()

streams = fetcher.run(urls=["https://example.com/"])["streams"]

converter = HTMLToDocument()
docs = converter.run(sources=streams)

generator = HuggingFaceLocalGenerator(model="numind/NuExtract", huggingface_pipeline_kwargs={"model_kwargs": {"torch_dtype":torch.float32}})

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
print(str(result))

extracted_data = result["replies"][0]
with open('output.json', 'w') as f:
    json.dump(extracted_data, f, indent=4)