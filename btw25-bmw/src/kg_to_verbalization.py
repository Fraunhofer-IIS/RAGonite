import json
import os
from typing import Optional, List, Any, Dict, Union
from pathlib import Path
import asyncio

from ragonite.data import DocumentDatabase, Document
from ragonite.embeddings import embed_documents, EmbeddingModel

import re
from urllib.parse import urlparse

triple_pattern = re.compile(r'<([^>]+)>\s*a\s*([^;]+)\s*;(.*?)(?=\.\s*<|$)', re.DOTALL)
properties_pattern = re.compile(r'(\S+)\s+("[^"]*"|<[^>]*>|[^;]+)\s*;', re.DOTALL)


def analyze_triples(triples: List[Dict[str, Any]]):
    total_property_assertions = 0
    total_type_assertions = 0
    total_literals = 0

    unique_predicates = set()
    unique_types = set()
    unique_entities = set()

    for entity in triples:
        original_uri = entity.get('original_uri')
        unique_entities.add(original_uri)

        properties = entity.get('properties', {})
        types = entity.get('types', [])

        for predicate, values in properties.items():
            unique_predicates.add(predicate)

            if not isinstance(values, list):
                values = [values]

            num_values = len(values)
            total_property_assertions += num_values

            for value in values:
                if isinstance(value, (str, int, float)):
                    total_literals += 1

        if not isinstance(types, list):
            types = [types]

        unique_types.update(types)
        num_types = len(types)
        total_type_assertions += num_types

    total_facts = total_property_assertions + total_type_assertions
    total_entities = len(unique_entities)
    unique_predicates_count = len(unique_predicates)
    unique_types_count = len(unique_types)

    print(10 * "-" + " Stats " + 10 * "-")
    print(f"Dataset: Unique predicates: {unique_predicates_count}")
    print(f"- These are: {unique_predicates}")
    print(f"Dataset: Unique types: {unique_types_count}")
    print(f"- These are: {unique_types}")
    print(f"Num entities: {total_entities}")
    print(f"Num facts: {total_facts}")
    print(f"Num property assertions: {total_property_assertions}")
    print(f"Num type assertions: {total_type_assertions}")
    print(f"Num literals: {total_literals}")


def parse_turtle(file_content: str):
    triples = []
    triple_pattern = re.compile(r'(<[^>]+>)\s+a\s+([^;]+);\s+(.*?)(?=\.\s*<|$)', re.DOTALL)

    for triple_match in triple_pattern.finditer(file_content):
        subject = triple_match.group(1).strip()
        types = [t.strip() for t in triple_match.group(2).split(',')]
        properties_block = triple_match.group(3).strip()

        properties = {}

        properties_pattern = re.compile(r'ns1:(\w+)\s+(?:"([^"]+)"|<([^>]+)>)\s*[;,]?')

        current_predicate = None

        for line in properties_block.splitlines():
            line = line.strip()

            prop_match = properties_pattern.match(line)
            if prop_match:
                predicate = prop_match.group(1).strip()
                object_value = prop_match.group(2) or prop_match.group(3)

                if object_value and object_value.startswith("<") and object_value.endswith(">"):
                    object_value = object_value[1:-1]  # Remove angle brackets from link objects

                if predicate in properties:
                    properties[predicate] += f', {object_value}'
                else:
                    properties[predicate] = object_value

                current_predicate = predicate
            else:
                # Handle continuation lines
                if current_predicate:
                    continuation_match = re.match(r'^\s*"([^"]+)"\s*[;,]?', line)
                    if continuation_match:
                        object_value = continuation_match.group(1).strip()
                        properties[current_predicate] += f'[AND] {object_value}'
                    else:
                        continuation_match = re.findall(r'<([^>]+)>', line)
                        for match in continuation_match:
                            match = match.strip()
                            if match not in properties[current_predicate]:
                                properties[current_predicate] += f', {match}'

        triple_dict = {
            'subject': subject[1:-1],
            'types': types,
            'properties': properties
        }
        triples.append(triple_dict)

    return triples


def handle_special_props(prop: str):
    if prop == "speed":
        return "topSpeed"
    else:
        return prop


def handle_special_values(prop: str, value: str):
    """
    Normalize floating points to US number format, convert mm to meters, and handle special hyphen cases.
    """
    # Normalize floating-point numbers to US format
    # Remove the European thousands separator (dot) and replace the comma with a dot
    value = re.sub(r'(\d+)\.(\d{3})', r'\1\2', value)  # Removes thousands dot
    value = re.sub(r'(\d+),(\d+)', r'\1.\2', value)  # Replaces decimal comma with dot

    # Convert mm to meters
    def convert_mm_to_m(match):
        number = float(match.group(1)) / 1000  # Convert mm to meters
        return f'{number:.3f} m'  # Format with 3 decimal places

    value = re.sub(r'(\d+(\.\d+)?)\s*mm', convert_mm_to_m, value)

    # Replace hyphens in ranges with 'to'
    value = re.sub(r'(\b\d+)\s*-\s*(\d+\b)', r'\1 to \2', value)

    # Remove single hyphens before units
    value = re.sub(r'\s*-\s*([a-zA-Z/]+)', r' \1', value)

    # Handle multiple values as ranges
    value = re.sub(r'"([^"]+)",\s*"([^"]+)"', r'\1 to \2', value)
    value = re.sub(r'"([^"]+)"\s*,\s*\n\s*"([^"]+)"', r'\1 to \2', value)

    # Append "Euros" for price properties
    if "price" in prop:
        value += " Euros"

    if "height" in prop or "width" in prop or "wheelbase" in prop or "length" in prop:
        value += " mm"

    return value


def verbalize_url(url: str):
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')

    if 'instance' in path_parts:
        instance_index = path_parts.index('instance')
        path_parts = path_parts[instance_index + 1:]

    if len(path_parts) >= 2:
        category = path_parts[-2]
        subject = path_parts[-1]

        category = category.replace('-', ' ')
        subject = subject.replace('-', ' ')

        category = ' '.join([word.capitalize() for word in category.split()])
        subject = ' '.join([word.capitalize() for word in subject.split()])

        if len(path_parts) > 2:
            preceding_category = path_parts[-3].replace('-', ' ')
            preceding_category = ' '.join([word.capitalize() for word in preceding_category.split()])
            category = f"{category} {preceding_category}"

        verbalization = f"{subject} is {category}."

        return verbalization
    return None


def find_subject(triples: List[Dict[str, Any]], uris: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Find triples with the given URIs as the subject.
    """
    if isinstance(uris, str):
        uris = [uris]
    uris = [uri.strip('<>') for uri in uris]

    found_triples = []
    for uri in uris:
        for triple in triples:
            if triple["original_uri"] == uri:
                found_triples.append(triple)
                break
    return found_triples


def extract_subject_from_uri(triple: Dict[str, Any]) -> str:
    path = urlparse(triple["subject"]).path
    subject_name = path.rstrip('/').split('/')[-1]
    subject_name = subject_name.replace('-', ' ')
    subject_name = subject_name.replace('+', ' ')
    subject_name = subject_name.replace('%', ' ')

    return subject_name


def split_multiple_values(text: str):
    """
    Split properties with multiple values into separate statements.
    """
    values = re.split(r',\s*\n\s*', text)
    return values


def preprocess_triples(triples: List[Dict[str, Any]]):
    """
    Preprocess triples by replacing <URI> with label wherever possible.
    """
    for triple in triples:
        triple["original_uri"] = triple["subject"]
        label = triple["properties"].get("rdfs:label") or triple["properties"].get("ns1:name")
        if label:
            triple["subject"] = label
        else:
            subject_name = extract_subject_from_uri(triple)
            triple["subject"] = subject_name

    return triples


def camel_case_to_sentence(camel_case: str) -> str:
    """
    Convert a camel case string to a sentence with spaces and lowercase,
    handling special cases for strings with numbers like CO2.
    """
    s = re.sub(r'(\d+)', r' \1 ', camel_case)
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return ' '.join(s.split()).lower()


def pascal_case_to_sentence(pascal_case: str) -> str:
    """Convert a Pascal case string to a sentence with spaces and proper capitalization."""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', pascal_case)
    return s


def special_case_format(prop: str) -> str:
    if 'CO2' in prop:
        parts = re.split(r'(CO2)', prop)
        parts = [part if part != 'CO2' else 'CO2' for part in parts]
        return ' '.join(parts)
    return prop


def remove_prefix(prop: str) -> str:
    """
    Remove the prefix from a property name.
    e.g.: ns1:Car -> Car
    """
    return prop.split(':', 1)[-1]


def clean_escape_characters(text: str) -> str:
    text = text.replace('\\"', '"')
    text = text.replace('\"', '')
    text = re.sub(r'(?<!\d)\s+"', '', text)
    return text


def clean_property_string(property_string: str) -> str:
    # Replace + between word characters or digits with a space
    cleaned_string = re.sub(r'(?<=\w)\+(?=\w)', ' ', property_string)
    return cleaned_string


def create_property_statements(subject: str, prop_name: str, values: List[str], reverse_facts: bool) -> List[str]:
    statements = []
    for value in values:
        prop_sentence = f"{prop_name} {value}"
        property_string = f"{subject} has {prop_sentence}."
        if reverse_facts:
            if prop_name.endswith(" of"):
                prop_name = prop_name[:-3]
            if prop_name.startswith("is "):
                prop_name = prop_name[3:]
            property_string += f" {value} is {prop_name} of {subject}."
        statements.append(property_string)
    return statements


def format_prop_name(prop: str) -> str:
    prop_name = remove_prefix(prop)
    prop_name = special_case_format(prop_name)
    if re.match(r'[a-z]+([A-Z][a-z]*)+', prop_name):
        return camel_case_to_sentence(prop_name)
    return prop_name.replace('_', ' ')


def extract_nested_labels(nested_triples: List[Dict[str, Any]]) -> List[str]:
    nested_labels = []
    for nested_triple in nested_triples:
        label = nested_triple["properties"].get("rdfs:label") or nested_triple["properties"].get("ns1:name") or extract_subject_from_uri(nested_triple)
        if label:
            nested_labels.append(label)
    return nested_labels


def clean_value(prop: str, value: str) -> str:
    value = handle_special_values(prop, value)
    return clean_escape_characters(value)


def verbalize_triples(
    triples: List[Dict[str, Any]],
    reverse_facts: Optional[bool] = False,
) -> Dict[str, Any]:

    verbalizations = {}

    for triple in triples:
        original_uri = triple["original_uri"]
        subject = triple["subject"]
        types = triple.get("types", [])
        properties = triple.get("properties", {})

        if subject not in verbalizations:
            verbalizations[original_uri] = {
                "verbalizations": [],
                "id": original_uri,
                "title": subject,
                "url": properties.get("ns1:url", "")
            }

        if types:
            type_statements = [f"{subject} is a {pascal_case_to_sentence(remove_prefix(t))}" for t in types]
            verbalizations[original_uri]["verbalizations"].append(". ".join(type_statements) + ".")

        uri_pattern = re.compile(r'<?https?://[^,>\s]+>?')
        for prop, vals in properties.items():
            prop = handle_special_props(prop)
            if prop is None:
                continue

            prop_name = format_prop_name(prop)
            values = split_multiple_values(vals)

            for value in values:
                if re.match(r'<?https?://', value):
                    uris = re.findall(uri_pattern, value)
                    nested_triples = find_subject(triples, [uri.strip('<>') for uri in uris])
                    nested_labels = extract_nested_labels(nested_triples)
                    verbalizations[original_uri]["verbalizations"].extend(
                        create_property_statements(subject, prop_name, nested_labels, reverse_facts)
                    )
                else:
                    value = clean_value(prop, value)
                    if "[AND]" in value:
                        values = [v.strip() for v in value.split("[AND]")]
                    else:
                        values = [value]
                    verbalizations[original_uri]["verbalizations"].extend(
                        create_property_statements(subject, prop_name, values, reverse_facts)
                    )

    return verbalizations


def parse_additional_texts(json_file: Path) -> List[Document]:
    with open(json_file, "r") as f:
        data = json.load(f)

    documents = []

    for entry in data:
        url = entry["url"]
        texts = entry["texts"]

        incrementor = 1

        match = re.search(r'/bmw-([^/]+)/([^/]+)/', url, re.IGNORECASE)
        if match:
            model = f"BMW {match.group(2).replace('-', ' ').title()}"
        else:
            model = "BMW"

        for text in texts:
            doc_id = f"{url}_{incrementor}"
            incrementor += 1

            document = Document(
                id=doc_id,
                title=model,
                content=text,
                url=url
            )

            documents.append(document)

    return documents


def setup(
    database: DocumentDatabase,
    ttl_file: Optional[Path] = None,
    texts_json_file: Optional[Path] = None,
    output_file: Optional[Path] = "data/graph-digest/bmw-verbalizations.txt",
    reverse_facts: Optional[bool] = False,
):

    if not ttl_file and not texts_json_file:
        raise Exception("Specify either an input file (Knowledge Graph), or a Text File, or both.")

    if ttl_file:
        with open(ttl_file, 'r') as file:
            file_content = file.read()

        triples = parse_turtle(file_content)

        triples = preprocess_triples(triples)
        verbalizations = verbalize_triples(triples, reverse_facts=reverse_facts)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as file:
            json.dump(verbalizations, file, indent=4)

    else:
        verbalizations = None

    async def main():
        if verbalizations:
            for key in verbalizations:
                document = Document(
                    id=verbalizations[key]["id"],
                    title=verbalizations[key]["title"],
                    content=" ".join(verbalizations[key]["verbalizations"]),
                    url=verbalizations[key]["url"],
                )
                print(document)

                await embed_documents([document], database.embedding_model)
                await database.add_documents([document])

        if texts_json_file:
            print("Now embedding additional text documents.")
            additional_documents = parse_additional_texts(texts_json_file)
            for additional_document in additional_documents:
                print(additional_document)
                await embed_documents([additional_document], database.embedding_model)
                await database.add_documents([additional_document])

        if triples:
            analyze_triples(triples)

    asyncio.run(main())


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(setup, as_positional=False)