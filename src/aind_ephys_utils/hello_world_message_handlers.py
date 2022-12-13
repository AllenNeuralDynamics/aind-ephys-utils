""" Module to handle printing messages to stdout.
"""

import logging

import pandas as pd


class MessageHandler:
    """
    Class to handle messages.
    """

    def __init__(self, msg: str) -> None:
        """
        Initializes a message handler.
        Parameters
        ----------
        msg : str
          The message to handle.
        """
        self.msg = msg

    def log_msg(self):
        """Simply logs the message."""
        logging.info(self.msg)

    def msg_as_df(self, col_name: str = "message") -> pd.DataFrame:
        """
        Converts the message to a pandas dataframe
        Parameters
        ----------
        col_name : str
          A column name for the DataFrame. Default is 'message'.

        Returns
        -------
        pd.DataFrame
          A DataFrame with column name col_name and content from message.
        """
        return pd.DataFrame.from_dict({col_name: [self.msg]})
