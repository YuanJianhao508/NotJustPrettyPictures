import re
import importlib
from radar_plot import plot_radar, plot_radar_sbs

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


dataset = "officehome"
model = 50
experiment = 2

module_name = f"Results.{dataset}"
module = importlib.import_module(module_name)

attribute_name1 = f"{dataset.upper()}_18_{experiment}"
latex_table1 = getattr(module, attribute_name1)

attribute_name2 = f"{dataset.upper()}_50_{experiment}"
latex_table2 = getattr(module, attribute_name2)

# print(latex_to_dict_single_value(latex_table))

data_sample1 = latex_to_dict_single_value(latex_table1)
data_sample2 = latex_to_dict_single_value(latex_table2)

# plot_radar(data_sample,save_path=f"{dataset}_ResNet{model}_{experiment}.png")

plot_radar_sbs(data_sample1, data_sample2,save_path=f"{dataset}_{experiment}_sbs_2.png")