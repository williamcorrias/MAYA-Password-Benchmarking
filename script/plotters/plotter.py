import os

from script.utils.file_operations import redirect_stdout, reset_stderr, reset_stdout

def _get_csv_fieldnames(path):
    with open(path, "r") as f:
        fieldnames = f.readline().strip().split(",")
    return fieldnames


class Plotter:
    def __init__(self, rows, settings, test_name, weights):
        self._prepare_environment(settings, test_name, weights)
        self._prepare_plot_settings()
        self._prepare_logging()

        self.data = {}
        self.weights = {} if weights else None

        self._load_data(rows)
        self._sort_data()
        self._plot_data()
        self._extra()

    def _prepare_environment(self, settings, test_name, weights):
        self.settings = settings
        self.test_name = test_name
        self.weights = weights

    def _prepare_plot_settings(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def _prepare_logging(self):
        reset_stderr()
        reset_stdout()
        self.dest_folder = os.path.join("figures", self.test_name)
        os.makedirs(self.dest_folder, exist_ok=True)
        self.output_file = os.path.join(self.dest_folder, "output.txt")
        redirect_stdout(self.output_file)

    def _load_data(self, rows):
        return self._parse_rows(rows)

    def _parse_rows(self, rows):
        for path, row_list in rows.items():
            with open(path, "r") as f:
                headers = f.readline().strip().split(",")

            for row_str in row_list:
                row_values = row_str.strip().split(",")
                row = dict(zip(headers, row_values))
                self._process_single_row(row)

    def _process_single_row(self, row):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def _sort_data(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def _plot_data(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def _extra(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def _fix_logging(self):
        reset_stderr()
        reset_stdout()
        redirect_stdout(self.output_file)