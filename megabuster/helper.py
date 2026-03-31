import numpy as np
import pandas as pd 
from cmbdb import cmbdb

def get_instrument(tag=''):
    """ Get a pre-defined instrumental configuration

    Parameters
    ----------
    tag: string
        name of the pre-defined experimental configurations.
        It can contain the name of multiple experiments separated by a space.
        Call the function with a random input to get the available instruments.

    Returns
    -------
    instr: pandas.DataFrame
        It contains the experimental configuration of the desired instrument(s).

    Note
    ----
    Function taken from https://github.com/CMBSciPol/MICMAC
    """
    df = cmbdb.loc[cmbdb['experiment'].isin(tag.split())]
    if df.empty:
        if tag == 'test':
            df = pd.DataFrame()
            df['frequency'] = np.arange(10., 300, 30.)
            df['depth_p'] = (np.linspace(20, 40, 10) - 30)**2
            df['depth_i'] = (np.linspace(20, 40, 10) - 30)**2
        else:
            from importlib.util import find_spec
            exp_file = find_spec('cmbdb').submodule_search_locations[0]
            exp_file += '/experiments.yaml'
            github = 'https://github.com/dpole/cmbdb'
            raise ValueError(
                (f"Instrument(s) {tag} not available." if tag else "") +
                f"Choose between: {' '.join(cmbdb.experiment.unique())}{_NL}"
                f"Add your instrument to your local copy of cmbdb: {exp_file}\n"
                f"Beware, you might lose changes when you update: "
                f"push your new configuration to {github}")
    return df.dropna(axis=1, how='all')
