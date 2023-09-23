import re
import importlib
from radar_plot import plot_radar, plot_radar_sbs
from bar_plot import plot_bar

def latex_to_dict_single_value(latex_table):
    # Split the table into lines
    lines = latex_table.strip().split('\\')

    # Extract headers
    headers = [header.strip() for header in lines[0].split('&')[1:]]

    # Initialize the dictionary
    table_dict = {}

    for line in lines[1:]:  # Skip the header line
        # Split by '&' to separate method name from values
        parts = line.split('&')
        print(parts)
        if len(parts) < 2:
            continue

        method = parts[0].strip()
        values = parts[1:]

        # Extract numbers from the values
        method_values = {}
        for header, value in zip(headers, values):
            num = float(re.search(r'\d+\.\d+', value).group())
            method_values[header] = num

        table_dict[method] = method_values

    return table_dict


def extract_columns(texture_string):
    """
    Extracts the first and last columns from the provided texture string.

    Args:
    - texture_string (str): The input string in the given format.

    Returns:
    - tuple: Two lists representing the first and last columns.
    """
    # Splitting the string into lines and then into columns
    lines = texture_string.strip().split('\n')
    first_column = [line.split('&')[0].strip() for line in lines]
    last_column = [line.split('&')[-1].strip() for line in lines]

    # Removing unwanted characters and converting to float
    last_column = [float(value.replace('$', '').replace('\\', '').strip()) for value in last_column]

    return first_column, last_column


# dataset = "officehome"
# dataset = 'texture'
# dataset = 'celeba'
dataset = 'imagenet9'
model = 50
experiment = 1

if dataset in ['officehome','pacs','nico','domainnet']:
    module_name = f"Results.{dataset}"
    module = importlib.import_module(module_name)

    attribute_name1 = f"{dataset.upper()}_18_{experiment}"
    latex_table1 = getattr(module, attribute_name1)

    attribute_name2 = f"{dataset.upper()}_50_{experiment}"
    latex_table2 = getattr(module, attribute_name2)

    data_sample1 = latex_to_dict_single_value(latex_table1)
    data_sample2 = latex_to_dict_single_value(latex_table2)

    plot_radar_sbs(data_sample1, data_sample2,save_path=f"./Figs/{dataset}_{experiment}_sbs_2.png")

elif dataset in ['texture','imagenet9','celeba']:
    bias_mapping = {'texture': "Texture Bias", 'imagenet9': "Background Bias", 'celeba': "Demographic Bias"}
    module_name = f"Results.{dataset}"
    module = importlib.import_module(module_name)
    attribute_name1 = f"{dataset.upper()}"
    latex_table = getattr(module, attribute_name1)
    methods, values = extract_columns(latex_table)
    print(methods, values)
    plot_bar(methods, values, title='Comparison of Methods', ylabel=bias_mapping[dataset], save_path=f"./Figs/{dataset}.png")