from IPython.core.magic import Magics, cell_magic, magics_class

@magics_class
class NotebookMagics(Magics):
    """Custom cell magics for controlling notebook execution by mode."""

    @staticmethod
    def _parse_modes(line):
        """Parse a comma-separated list of modes."""
        modes = {item.strip() for item in line.split(",") if item.strip()}

        if not modes:
            raise ValueError("At least one mode must be provided")

        return modes

    def _current_mode(self):
        """Return the current notebook mode."""
        if "mode" not in self.shell.user_ns:
            raise NameError("Notebook variable 'mode' is not defined")

        return str(self.shell.user_ns["mode"])

    @cell_magic
    def run_if_mode(self, line, cell):
        """
        Run the cell only when the current mode is one of the supplied modes.

        Example:
            %%run_if_mode DEVELOPMENT, DEBUG
        """
        allowed_modes = self._parse_modes(line)
        current_mode = self._current_mode()

        if current_mode not in allowed_modes:
            print(
                f"Skipped: mode={current_mode!r}, "
                f"allowed={sorted(allowed_modes)}"
            )
            return None
        
        self.shell.run_cell(cell)
        return None

    @cell_magic
    def skip_if_mode(self, line, cell):
        """
        Skip the cell when the current mode is one of the supplied modes.

        Example:
            %%skip_if_mode PRODUCTION, TEST
        """
        skipped_modes = self._parse_modes(line)
        current_mode = self._current_mode()

        if current_mode in skipped_modes:
            print(
                f"Skipped: mode={current_mode!r}, "
                f"excluded={sorted(skipped_modes)}"
            )
            return None

        self.shell.run_cell(cell)
        return None


def load_ipython_extension(ipython):
    """Register the custom notebook magics."""
    ipython.register_magics(NotebookMagics)