from typing import List, Dict, Any, Tuple
from bs4 import Tag, BeautifulSoup, NavigableString

from preprocessing.utils import truncate

def parse_table_headers(table: Tag) -> List[str]:
    '''
        Creates a map of headers for a given table.
        This function might seem overkill for something so simple, but for verbalization it is really important
        to get this right. Plus, we also support complex structures like multi-header tables:
        See https://openxt.atlassian.net/wiki/spaces/TEST/pages/761823271/OpenXT+9.0+Measurement+Test (accessed 07/2024) for an example

        Args:
            table (Tag): A BeautifulSoup Tag object representing
            the table element.

        Returns:
            List[str]: A list of strings, each entry corresponding to one
            table header at it's given column position.
    '''
    rows = table.find_all('tr')

    max_columns = 0
    for row in rows:
        col_count = sum(int(cell.get('colspan', 1)) for cell in row.find_all(['th', 'td']))
        max_columns = max(max_columns, col_count)

    # We need to keep track of the headers at each column index
    header_map = [[] for _ in range(max_columns)]

    # Here, we track which columns have been filled up to which row
    col_fill = [0] * max_columns

    for row in rows:
        cells = row.find_all(['th'])
        col_index = 0

        for cell in cells:
            while col_index < max_columns and col_fill[col_index] > 0:
                col_fill[col_index] -= 1
                col_index += 1

            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))
            cell_text = cell.get_text(separator=' ', strip=True)

            for i in range(colspan):
                if col_index + i < max_columns:
                    if not header_map[col_index + i]:
                        header_map[col_index + i].append(cell_text)
                    else:
                        header_map[col_index + i][-1] += ' ' + cell_text

                    # Mark how many rows this column will be filled for
                    col_fill[col_index + i] = max(col_fill[col_index + i], rowspan - 1)

            col_index += colspan

    final_headers = []

    for col_index in range(max_columns):
        combined_header = ' '.join(filter(None, header_map[col_index])).strip()
        final_headers.append(combined_header)

    return final_headers


def verbalize_table(table: Tag) -> List[str]:
    """
    Verbalizes the records of a table in a pattern like:
    header1 is value1, and header 2 is value2, etc.

    Args:
        table (Tag): A BeautifulSoup Tag object of the table.

    Returns:
        List[str]: A list of strings, each entry corresponding to one
        verbalized record from the table.
    """
    table_lines = []
    header = parse_table_headers(table)

    rows = table.find_all('tr')

    for row in rows:
        # Skip header rows because we already processed them
        if row.find_all(['th']):
            continue

        cells = row.find_all('td')
        if not cells:
            continue

        data = [cell.get_text(separator=' ', strip=True) for cell in cells]

        # Identify the last non-empty column
        non_empty_indices = [i for i, d in enumerate(data) if d.strip()]
        if not non_empty_indices:
            # Entire row is empty
            continue
        last_non_empty_col = non_empty_indices[-1]

        line_parts = []
        for col in range(last_non_empty_col + 1):
            cell_content = data[col]
            if header and col < len(header):
                segment = f"{header[col]} is {cell_content}"
            else:
                segment = cell_content

            if col == last_non_empty_col:
                segment += ".\n"
            else:
                segment += ", and "
            line_parts.append(segment)

        line = "".join(line_parts)
        table_lines.append(line)

    return table_lines


def html_table_to_markdown(html_content: str):
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')

    headers = []
    data = []
    column_widths = []

    for row in rows:
        cells = row.find_all(['th', 'td'])
        row_data = []
        for i, cell in enumerate(cells):
            text = cell.get_text(strip=True)
            row_data.append(text)

            # Update the maximum width of each column
            if len(column_widths) <= i:
                column_widths.append(len(text))
            else:
                column_widths[i] = max(column_widths[i], len(text))

        # If headers are already set, this row is data
        if headers:
            # Make sure each row has the same number of columns as headers
            while len(row_data) < len(headers):
                row_data.append("")
            if len(row_data) > len(headers):
                row_data = row_data[:len(headers)]
            data.append(row_data)
        else:
            # First row is considered headers
            headers = row_data

    # Ensure all rows in data have the same number of columns as headers
    for row in data:
        while len(row) < len(headers):
            row.append("")
        if len(row) > len(headers):
            row = row[:len(headers)]

    # Recalculate column widths based on headers and data
    column_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + data))]

    # Adjust column widths to ensure headers and all cells fit
    column_widths = [max(column_widths[i], len(headers[i])) for i in range(len(headers))]

    # Create the Markdown table format
    header_row = '| ' + ' | '.join(header.ljust(column_widths[i]) for i, header in enumerate(headers)) + ' |'
    separator_row = '| ' + ' | '.join('-' * column_widths[i] for i in range(len(headers))) + ' |'
    data_rows = [
        '| ' + ' | '.join(cell.ljust(column_widths[j]) for j, cell in enumerate(row)) + ' |'
        for row in data
    ]

    markdown_table = '\n'.join([header_row, separator_row] + data_rows)

    return markdown_table, [header_row, separator_row] + data_rows


def html_table_to_plaintext(html_content: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')

    plain_text_rows = []

    for row in rows:
        cells = row.find_all(['th', 'td'])
        row_text = '\t'.join(cell.get_text(strip=True) for cell in cells)
        plain_text_rows.append(row_text)

    plain_text_table = '\n'.join(plain_text_rows)

    return plain_text_table, plain_text_rows


def html_table_to_clean_html(html_content: str) -> Tuple[str, BeautifulSoup | NavigableString | None]:
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')

    if not table:
        raise ValueError("No table found in the provided HTML content.")

    clean_table = BeautifulSoup('<table></table>', 'html.parser').find('table')

    for row in table.find_all('tr'):
        clean_row = soup.new_tag('tr')
        for cell in row.find_all(['th', 'td']):
            tag_name = 'th' if cell.name == 'th' else 'td'
            clean_cell = soup.new_tag(tag_name)
            clean_cell.string = cell.get_text(strip=True)
            clean_row.append(clean_cell)
        clean_table.append(clean_row)

    clean_html = str(clean_table)
    return clean_html, clean_table


def html_table_to_piped(html_content: str, truncate_cells=True) -> Tuple[str, List[str]]:
    """
    Convert HTML table to a piped text format.

    Args:
        html_content (str): HTML content containing a table.
        truncate_cells (bool): If True, truncates cell content. Default is True.

    Returns:
        str: Table in piped text format.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')

    headers = []
    data = []
    column_widths = []

    for row in rows:
        cells = row.find_all(['th', 'td'])
        row_data = []
        for i, cell in enumerate(cells):
            text = cell.get_text(strip=True)

            if truncate_cells:
                text = truncate(text)

            row_data.append(text)

            # Adjust column widths
            if len(column_widths) <= i:
                column_widths.append(len(text))
            else:
                column_widths[i] = max(column_widths[i], len(text))

        if headers:
            # Add empty strings if row_data is shorter than headers
            while len(row_data) < len(headers):
                row_data.append("")
            # Truncate row_data if it's longer than headers
            if len(row_data) > len(headers):
                row_data = row_data[:len(headers)]
            data.append(row_data)
        else:
            headers = row_data
            # Add empty strings to column_widths if headers have more elements
            while len(column_widths) < len(headers):
                column_widths.append(0)

    # Ensure all rows in data have the same number of columns as headers
    for row in data:
        while len(row) < len(headers):
            row.append("")
        if len(row) > len(headers):
            row = row[:len(headers)]

    # Adjust column widths based on data rows
    column_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + data))]

    # Ensure column_widths and headers match in length
    if len(column_widths) != len(headers):
        print("Warning: headers and data columns do not match in length. Adjusting to match headers.")
        while len(column_widths) < len(headers):
            column_widths.append(0)
        if len(column_widths) > len(headers):
            column_widths = column_widths[:len(headers)]

    # Adjust column widths for headers if they are longer than any cell in that column
    column_widths = [max(column_widths[i], len(headers[i])) for i in range(len(headers))]

    # Add extra space for row counter
    row_counter_width = max(len(f"row_{len(data)}"), len("Row ID")) + 2  # Adjusted for uniform header alignment
    column_widths.insert(0, row_counter_width)

    header_row = ' | '.join(
        ['Row ID'.ljust(row_counter_width)] + [header.ljust(column_widths[i + 1]) for i, header in enumerate(headers)]
    )
    separator_row = '-+-'.join(['-' * row_counter_width] + ['-' * column_widths[i + 1] for i in range(len(headers))])

    data_rows = [
        f"{('row_' + f'{i + 1:02}').ljust(row_counter_width)} | " + ' | '.join(
            cell.ljust(column_widths[j + 1]) for j, cell in enumerate(row))
        for i, row in enumerate(data)
    ]

    text_table = '\n'.join([header_row, separator_row] + data_rows)

    return text_table, [header_row, separator_row] + data_rows


def table_to_json(element: Tag) -> Dict[str, Any]:
    headers = [cell.get_text(strip=True) for cell in element.find_all('th')]
    rows = []
    for row in element.find_all('tr'):
        cells = row.find_all(['td'])
        row_data = [cell.get_text(strip=True) for cell in cells]
        if row_data:
            rows.append(row_data)
    return {
        "headers": headers,
        "rows": rows
    }
