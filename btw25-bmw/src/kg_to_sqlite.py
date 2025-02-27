from typing import Union, Any, Dict, List, Literal as LiteralType, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
import re
import sqlite3

import rdflib


"""

Example Usage: python kg_to_sqlite.py --config config.yaml --input graph.ttl --output graph.db

The automatic SQL schema induction process works as follows:

The knowledge graph is loaded from a file in RDF Turtle/NTriples format and represented as a set of
S-P-O triples (subject, predicate, object) internally.

1. Preprocessing

First all triples are preprocessed. Triple filters can be configured that either drop triples for
certain predicates (e.g. due to redundancy), modify the object/value (e.g. removing units or
thousands seperators from numbers) or add additional triples (e.g. splitting a triple where the
object/value represents a range into two seperate triples, one for the minimum and one for the
maximum value).

2. Entity Grouping

Next all subject-predicate-object (S-P-O) triples are grouped by subject. The combination of all
predicates and objects that belong to the same subject is called an entity.

3. Entity Type Generation

Entities are grouped by type. The type of an entity is the determined by concatenating all
objects/values of the 22-rdf-syntax-ns#type predicate.

The shape of each entity type is analyzed next. All predicates of all entities that belong to the
type are collected and grouped by whether the corresponding objects/values are literal values (e.g
a number or a string) or point to another subject/entity.

4. Entity Type Relation Analysis

Then the relations between entity types are analzed. For each pair of entity types it is determined
if they have a 1:1, 1:N, N:1 or N:M relation (or no relation at all).

5. SQL Schema Generation and Data Population

Each entity type is mapped to a table in the SQLite database. The entity type name is used as the
table name.

One column is added for storing the subject/name of the entities. This column is used as the
primary key.

Then a column is added for each predicate connected to the entity type for which the object is a
literal. The datatype of this column is inferred based on the actual values of the literal objects.
For example if all values can be represented as an integer then the integer datatype is used for the
whole column.

Additional columns or tables are added to represent relations to other entities:

For an 1:N (or 1:1) relation between entity types A and B a column is added to table B that points
to a row of table A. For N:1 relations this is done the other way around.

For N:M relations an additional table is added, that contains two columns: one with the foreign keys
to table A and one with foreign keys to table B.

Each entity is inserted as a row into the table of the corresponding entity type. In case of a N:M
relation an additional row is added to the table that represents the relation.

6. SQL Schema Postprocessing

Finally postprocessing can be applied to the inferred schema in order to improve the readability and
making it easier to understand for the model. Based on a configuration from the user tables and
columns can be renamed, redundant columns can be dropped and comments can be added to table columns,
e.g. to specify the unit or give example values.

The final output of this tool is a single SQLite database file that contains all data represented
according to the inferred schema.

"""


def print_table(items: List[Any], keys: List[str], max_column_width: int = 1000):
    columns = [[] for key in keys]

    for item in items:
        for i, key in enumerate(keys):
            if isinstance(item, dict):
                value = item.get(key, None)
            else:
                value = getattr(item, key)

            value = repr(value)
            value = value[0:max_column_width]
            columns[i].append(value)
    
    max_column_lengths = [max([len(cell) for cell in [*column, keys[i]]]) for i, column in enumerate(columns)]

    print("+-"+"-+-".join(["-" * max_column_lengths[i] for i, key in enumerate(keys)])+"-+")
    print("| "+" | ".join([key.ljust(max_column_lengths[i]) for i, key in enumerate(keys)])+" |")
    print("+-"+"-+-".join(["-" * max_column_lengths[i] for i, key in enumerate(keys)])+"-+")

    for i in range(len(items)):
        print("| "+" | ".join([column[i].ljust(max_column_lengths[k]) for k, column in enumerate(columns)])+" |")

    print("+-"+"-+-".join(["-" * max_column_lengths[i] for i, key in enumerate(keys)])+"-+")


@dataclass
class Literal:
    value: Any


@dataclass
class Link:
    link: str


Value = Union[Literal, Link]


@dataclass
class Entity:
    name: str
    type_name: str
    values: Dict[str, List[Value]]


@dataclass
class LiteralField:
    multi: bool = False


@dataclass
class LinkField:
    pass


@dataclass
class EntityType:
    name: str
    literals: Dict[str, LiteralField]
    links: Dict[str, LinkField]


def parse_german_number(value: str) -> str:
    value = value.replace(".", "")

    if "," in value:
        # some values had commas both as decimal seperator and thousands seperator
        [*a, b] = value.split(",")
        value = "".join(a)+"."+b

    value = str(float(value))
    value = re.sub(".0$", "", value)
    return value


def parse_time(value: str) -> int:
    values = value.split(":")
    return int(values[0]) * 60 + int(values[1])


def remove_unit(value: str, unit: str) -> str:
    expected_end = " "+unit.lower()
    
    if value.lower().endswith(expected_end):
        return value[:-len(expected_end)]
    else:
        return value


@dataclass
class ParseGermanNumberTripleFilter:
    predicates: List[str]
    type: LiteralType["parse_german_number"] = "parse_german_number"

    def run(self, predicate: str, value: str) -> Optional[List[Tuple[str, Value]]]:
        value = parse_german_number(value)
        return [(predicate, Literal(value=value))]


@dataclass
class RemoveUnitTripleFilter:
    predicates: List[str]
    unit: str
    remove_thousands_seperator: bool = False
    parse_german_number: bool = False
    parse_time: bool = False
    remove_prefix: Optional[str] = None
    type: LiteralType["remove_unit"] = "remove_unit"

    def run(self, predicate: str, value: str) -> Optional[List[Tuple[str, Value]]]:
        if self.remove_prefix is not None:
            if value.startswith(self.remove_prefix):
                value = value[len(self.remove_prefix):]

        if "(" in value and ")" in value:
            value = re.sub(r"\(.*?\)", "", value).strip()

        value = remove_unit(value, self.unit)

        if self.remove_thousands_seperator:
            value = value.replace(".", "")
        elif self.parse_german_number:
            value = parse_german_number(value)
        elif self.parse_time:
            value = parse_time(value)

        return [(predicate, Literal(value=value))]


@dataclass
class DropTripleFilter:
    predicates: List[str]
    type: LiteralType["drop"] = "drop"

    def run(self, predicate: str, value: str) -> Optional[List[Tuple[str, Value]]]:
        return []


@dataclass
class IgnoreValuesTripleFilter:
    predicates: List[str]
    values: List[str]
    type: LiteralType["ignore_values"] = "ignore_values"

    def run(self, predicate: str, value: str) -> Optional[List[Tuple[str, Value]]]:
        if value in self.values:
            return []
        else:
            return None


@dataclass
class SplitRangeTripleFilter:
    predicates: List[str]
    unit: Optional[str]
    parse_german_number: bool = True
    seperators: Optional[List[str]] = None
    type: LiteralType["split_range"] = "split_range"

    def run(self, predicate: str, value: str) -> Optional[List[Tuple[str, Value]]]:
        if value is None:
            return None

        if self.seperators is not None:
            seperators = self.seperators
        else:
            seperators = ["−", "-"] # utf8
        for seperator in seperators:
            value = value.replace(seperator, "||")
        values = [x.strip() for x in value.split("||")]

        if self.unit is not None:
            values = [remove_unit(value, self.unit) for value in values]

        if self.parse_german_number:
            values = sorted([parse_german_number(x) for x in values])
        else:
            values = sorted([x for x in values])

        if len(values) == 1:
            return [(predicate+"Min", Literal(value=values[0])), (predicate+"Max", Literal(value=values[0]))]
        else:
            return [(predicate+"Min", Literal(value=values[0])), (predicate+"Max", Literal(value=values[1]))]


TripleFilter = Union[ParseGermanNumberTripleFilter, RemoveUnitTripleFilter, DropTripleFilter, IgnoreValuesTripleFilter, SplitRangeTripleFilter]


@dataclass
class ConversionConfig:
    strip_uri_prefixes: List[str] = field(default_factory=list)
    triple_filters: List[TripleFilter] = field(default_factory=list)
    table_aliases: Dict[str, str] = field(default_factory=dict)
    relation_key_aliases: Dict[str, str] = field(default_factory=dict)
    table_field_comments: Dict[str, Dict[str, str]] = field(default_factory=dict)
    ignore_tables: List[str] = field(default_factory=list)
    ignore_table_fields: Dict[str, List[str]] = field(default_factory=dict)
    table_field_aliases: Dict[str, Dict[str, str]] = field(default_factory=dict)


def strip_name(name: str, config: ConversionConfig) -> str:
    for uri_prefix in config.strip_uri_prefixes:
        if name.startswith(uri_prefix):
            return name[len(uri_prefix):]

    return name


def strip_predicate(pred: str) -> str:
    pred = Path(pred).name

    if pred == "rdf-schema#label":
        return "label"

    return pred


def to_camel_case(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def uppercase_first(value: str) -> str:
    return value[0].upper() + value[1:]


def to_upper_camel_case(value: str) -> str:
    return uppercase_first(to_camel_case(value))


def is_integer_string(value: str) -> bool:
    try:
        int(value)
        return True
    except:
        return False


def is_float_string(value: str) -> bool:
    try:
        float(value)
        return True
    except:
        return False


def table_key_name(table_name: str) -> str:
    return to_camel_case(table_name)+"Id"


def process_triple_literal(key: str, value: str, config: ConversionConfig) -> List[Tuple[str, Value]]:
    for triple_filter in config.triple_filters:
        if key in triple_filter.predicates:
            result = triple_filter.run(key, value)
            if result is not None:
                return result

    return [(key, Literal(value=value))]


def process_triple_link(key: str, value: str, config: ConversionConfig) -> List[Tuple[str, Value]]:
    for triple_filter in config.triple_filters:
        if key in triple_filter.predicates:
            result = triple_filter.run(key, value)
            if result is not None:
                return result

    return [(key, Link(link=value))]


def process_triple(sub, pred, obj, config: ConversionConfig) -> List[Tuple[str, Value]]:
    if isinstance(obj, rdflib.term.URIRef):
        return process_triple_link(pred, strip_name(str(obj), config), config)
    elif isinstance(obj, rdflib.term.Literal):
        return process_triple_literal(pred, str(obj), config)
    else:
        raise Exception("invalid object type")


def get_entity_type_name(entity: Entity, config: ConversionConfig) -> str:
    type_name = "-".join(sorted([l.link for l in entity.values["22-rdf-syntax-ns#type"]]))

    if type_name in config.table_aliases:
        return config.table_aliases[type_name]
    else:
        return type_name


def get_entities_from_graph(graph: rdflib.Graph, config: ConversionConfig) -> List[Entity]:
    entities = {}

    # get rid of rdflib's randomization of the triple order
    triples = sorted([(sub, pred, obj) for sub, pred, obj in graph])

    for sub, pred, obj in triples:
        sub = strip_name(str(sub), config)
        pred = strip_predicate(str(pred))

        if sub not in entities:
            entities[sub] = Entity(name=sub, type_name="", values={})

        for key, value in process_triple(sub, pred, obj, config):
            if key not in entities[sub].values:
                entities[sub].values[key] = []

            entities[sub].values[key].append(value)

    for entity in entities.values():
        entity.type_name = get_entity_type_name(entity, config)

    return entities


def create_entity_types(entities: List[Entity]) -> Dict[str, EntityType]:
    entity_types = {}

    for entity in entities.values():
        if entity.type_name not in entity_types:
            entity_types[entity.type_name] = EntityType(name=entity.type_name, literals={}, links={})

        for key, values in entity.values.items():
            if key == "22-rdf-syntax-ns#type":
                continue

            literal_values = 0
            link_values = 0

            for value in values:
                if isinstance(value, Literal):
                    literal_values += 1
                elif isinstance(value, Link):
                    link_values += 1

            if literal_values > 0 and link_values > 0:
                raise Exception(f"field {key} has literals and relations to other entities as values")

            if literal_values > 0:
                if key not in entity_types[entity.type_name].literals:
                    entity_types[entity.type_name].literals[key] = LiteralField()

                if literal_values > 1:
                    entity_types[entity.type_name].literals[key].multi = True

            if link_values > 0:
                if key not in entity_types[entity.type_name].links:
                    entity_types[entity.type_name].links[key] = LinkField()


    return entity_types


@dataclass
class Relation:
    type: LiteralType["1-1", "1-N", "N-1", "N-M"]
    type_a: str
    key: str
    type_b: str


def create_relations(entities: Dict[str, Entity], entity_types: Dict[str, EntityType]) -> List[Relation]:
    relations = {}

    # forward
    reverse_links = {}
    for entity in entities.values():
        for key, values in entity.values.items():
            if key == "22-rdf-syntax-ns#type" or key == "rdf-schema#label":
                continue

            key_relations = {}
            for value in values:
                if isinstance(value, Link):
                    target = entities[value.link]

                    if target.type_name not in key_relations:
                        key_relations[target.type_name] = 1
                    else:
                        key_relations[target.type_name] += 1

                    if value.link not in reverse_links:
                        reverse_links[value.link] = {}
                    if key not in reverse_links[value.link]:
                        reverse_links[value.link][key] = []
                    reverse_links[value.link][key].append(entity.name)

            for target_type, count in key_relations.items():
                index = (entity.type_name, key, target_type)
                if index in relations:
                    relations[index] = (relations[index][0], max(relations[index][1], count))
                else:
                    relations[index] = (1, count)

    # backward
    for entity_name in reverse_links.keys():
        for key, links in reverse_links[entity_name].items():
            key_relations = {}

            for link in links:
                target = entities[link]
                if target.type_name not in key_relations:
                    key_relations[target.type_name] = 1
                else:
                    key_relations[target.type_name] += 1

            for target_type, count in key_relations.items():
                index = (target_type, key, entities[entity_name].type_name)
                if index in relations:
                    relations[index] = (max(relations[index][0], count), relations[index][1])
                else:
                    relations[index] = (count, 1)

    result = []
    for index, (a, b) in relations.items():
        if a > 1 and b > 1:
            result.append(Relation(type="N-M", type_a=index[0], key=index[1], type_b=index[2]))
        elif b > 1:
            result.append(Relation(type="1-N", type_a=index[0], key=index[1], type_b=index[2]))
        elif a > 1:
            result.append(Relation(type="N-1", type_a=index[0], key=index[1], type_b=index[2]))
        elif a == 1 and b == 1:
            result.append(Relation(type="1-1", type_a=index[0], key=index[1], type_b=index[2]))
        else:
            raise Exception("unreachable")

    return result


@dataclass
class TableField:
    nullable: bool = False
    type: Optional[LiteralType["INTEGER", "REAL", "TEXT"]] = None
    comment: Optional[str] = None


@dataclass
class TableForeignKey:
    field_name: str
    table_name: str
    table_primary_key: str


@dataclass
class Table:
    name: str
    primary_key: Optional[str]
    fields: Dict[str, TableField]
    foreign_keys: List[TableForeignKey]
    rows: Dict[str, Dict[str, Any]]


def get_relation_key(relation: Relation, config: ConversionConfig, reverse: bool = False) -> str:
    if not reverse:
        key = to_camel_case(relation.type_a)+"To"+to_upper_camel_case(relation.type_b)
    else:
        key = to_camel_case(relation.type_b)+"To"+to_upper_camel_case(relation.type_a)

    if key in config.relation_key_aliases:
        return config.relation_key_aliases[key]
    else:
        return key


def create_tables(entities, entity_types, relations, config: ConversionConfig) -> Dict[str, Table]:
    tables = {}

    # create tables and insert values
    for entity_type in entity_types.values():
        table = Table(
            name=entity_type.name,
            primary_key=table_key_name(entity_type.name),
            fields={},
            foreign_keys=[],
            rows={},
        )
        tables[entity_type.name] = table

        for literal_name, literal in entity_type.literals.items():
            if literal.multi:
                raise Exception(f"unimplemented multi literal {literal_name}")

            table.fields[literal_name] = TableField()

        for relation in relations:
            if relation.type == "1-N" and relation.type_b == entity_type.name:
                # 1-N example: 1 author n articles, relation=hasWritten
                table.fields[get_relation_key(relation, config, reverse=True)] = TableField()
                table.foreign_keys.append(TableForeignKey(
                    field_name=get_relation_key(relation, config, reverse=True),
                    table_name=relation.type_a,
                    table_primary_key=table_key_name(relation.type_a),
                ))
            elif (relation.type == "N-1" or relation.type == "1-1") and relation.type_a == entity_type.name:
                # N-1 example: n articles 1 author, relation=authoredBy
                table.fields[get_relation_key(relation, config)] = TableField()
                table.foreign_keys.append(TableForeignKey(
                    field_name=get_relation_key(relation, config),
                    table_name=relation.type_b,
                    table_primary_key=table_key_name(relation.type_b),
                ))

        for entity in entities.values():
            if entity.type_name == entity_type.name:
                row = {}

                row[table_key_name(entity_type.name)] = entity.name

                for literal_name in entity_type.literals.keys():
                    if literal_name in entity.values:
                        row[literal_name] = entity.values[literal_name][0].value

                table.rows[entity.name] = row

    # insert values for relations
    for relation in relations:
        if relation.type == "1-N":
            # 1-N example: 1 author n articles, relation=hasWritten
            for entity_a in entities.values():
                if entity_a.type_name == relation.type_a:
                    for target_value in entity_a.values.get(relation.key, []):
                        target = entities[target_value.link]
                        tables[target.type_name].rows[target.name][get_relation_key(relation, config, reverse=True)] = entity_a.name

        elif relation.type == "N-1" or relation.type == "1-1":
            # N-1 example: n articles 1 author, relation=authoredBy
            for entity_a in entities.values():
                if entity_a.type_name == relation.type_a:
                    for target_value in entity_a.values.get(relation.key, []):
                        target = entities[target_value.link]
                        tables[relation.type_a].rows[entity_a.name][get_relation_key(relation, config)] = target.name
        elif relation.type == "N-M":
            table = Table(
                name=relation.type_a+"_to_"+relation.type_b,
                primary_key=None,
                fields={
                    table_key_name(relation.type_a): TableField(),
                    table_key_name(relation.type_b): TableField(),
                },
                foreign_keys=[
                    TableForeignKey(
                        field_name=table_key_name(relation.type_a),
                        table_name=relation.type_a,
                        table_primary_key=table_key_name(relation.type_a),
                    ),
                    TableForeignKey(
                        field_name=table_key_name(relation.type_b),
                        table_name=relation.type_b,
                        table_primary_key=table_key_name(relation.type_b),
                    ),
                ],
                rows={},
            )
            tables[table.name] = table

            for entity_a in entities.values():
                if entity_a.type_name == relation.type_a:
                    for target_value in entity_a.values.get(relation.key, []):
                        target = entities[target_value.link]
                        table.rows[entity_a.name+"-"+target.name] = {
                            table_key_name(relation.type_a): entity_a.name,
                            table_key_name(target.type_name): target.name,
                        } 

    # convert column data types
    for table in tables.values():
        for field_name, field in table.fields.items():
            nullable = False
            has_integers = False
            has_floats = False
            has_strings = False

            for row in table.rows.values():
                field_value = row.get(field_name, None)
                
                if field_value is None:
                    nullable = True
                elif is_integer_string(field_value):
                    has_integers = True
                elif is_float_string(field_value):
                    has_floats = True
                else:
                    has_strings = True

            if has_strings:
                field_type = "TEXT"
            elif has_floats:
                field_type = "REAL"
            elif has_integers:
                field_type = "INTEGER"
            else:
                field_type = "TEXT"

            field.type = field_type
            field.nullable = nullable

            if field_type == "REAL":
                for row in table.rows.values():
                    field_value = row.get(field_name, None)
                    if field_value is not None:
                        row[field_name] = float(field_value)
            elif field_type == "INTEGER":
                for row in table.rows.values():
                    field_value = row.get(field_name, None)
                    if field_value is not None:
                        row[field_name] = int(field_value)

    return tables


def create_table_schema(table: Table) -> str:
    """
        CREATE TABLE {{ table.name }} (
        {%- if table.primary_key %}
            {{ table.primary_key }} TEXT PRIMARY KEY,
        {%- endif %}
        {%- for field_name, field in table.fields.items() %}
            {{ field_name }} {{ field.type }}{% if not field.nullable %} NOT NULL{% endif %}{% if not loop.last or table.foreign_keys %},{% endif %}{% if field.comment %} -- {{ field.comment }}{% endif %}
        {%- endfor %}
        {%- for foreign_key in table.foreign_keys %}
            FOREIGN KEY ({{ foreign_key.field_name }}) REFERENCES {{ foreign_key.table_name}}({{ foreign_key.table_primary_key }})
        {%- endfor %}
        )
    """

    output = [f"CREATE TABLE {table.name} ("]
    
    if table.primary_key:
        output.append(f"\t{table.primary_key} TEXT PRIMARY KEY,")
    
    for i, (field_name, field) in enumerate(table.fields.items()):
        line = f"\t{field_name} {field.type}"
        if not field.nullable:
            line += " NOT NULL"
        
        if i < len(table.fields) - 1 or table.foreign_keys:
            line += ","
        
        if field.comment:
            line += f" -- {field.comment}"
        
        output.append(line)

    for i, foreign_key in enumerate(table.foreign_keys):
        output.append(f"\tFOREIGN KEY ({foreign_key.field_name}) REFERENCES {foreign_key.table_name}({foreign_key.table_primary_key})")

    output.append(")")
    return "\n".join(output)


def persist_tables(tables: List[Table], filename: Path):
    conn = sqlite3.connect(filename)

    cursor = conn.cursor()

    for table in tables:
        cursor.execute(create_table_schema(table))

        for row in table.rows.values():
            keys = [*row.keys()]
            values = [*row.values()]
            cursor.execute(f"INSERT INTO {table.name} ({', '.join(keys)}) VALUES ({', '.join(['?' for value in values])})", values)

    conn.commit()


def convert_kg_to_sqlite(
    input: Path,
    output: Path,
    conversion: ConversionConfig,
):
    graph = rdflib.Graph()
    graph.parse(input, format="ttl")

    # 1. Preprocessing and 2. Entity Grouping
    entities = get_entities_from_graph(graph, conversion)

    # 3. Entity Type Generation
    entity_types = create_entity_types(entities)

    # 4. Entity Type Relation Analysis
    relations = create_relations(entities, entity_types)
    pprint(relations)

    # 5. SQL Schema Generation and Data Population
    tables = create_tables(entities, entity_types, relations, conversion)

    # 6. SQL Schema Postprocessing
    for table_name, aliases in conversion.table_field_aliases.items():
        for field_name, alias in aliases.items():
            field = tables[table_name].fields[field_name]
            del tables[table_name].fields[field_name]
            tables[table_name].fields[alias] = field

            for row_id, row in tables[table_name].rows.items():
                if field_name in row:
                    value = row[field_name]
                    del row[field_name]
                    row[alias] = value

    for table_name, fields in conversion.ignore_table_fields.items():
        for field_name in fields:
            del tables[table_name].fields[field_name]

            for row_id, row in tables[table_name].rows.items():
                if field_name in row:
                    del row[field_name]

    for table_name, comments in conversion.table_field_comments.items():
        for field_name, comment in comments.items():
            tables[table_name].fields[field_name].comment = comment

    for table_name in conversion.ignore_tables:
        del tables[table_name]

    for table in tables.values():
        print(create_table_schema(table))

        if table.primary_key is not None:
            print_table(table.rows.values(), [table.primary_key, *table.fields.keys()], max_column_width=30)
        else:
            print_table(table.rows.values(), [*table.fields.keys()], max_column_width=30)
        
        print()

    # creating a sqlite database with the inferred schema and data
    persist_tables(tables.values(), output)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(convert_kg_to_sqlite, as_positional=False)
