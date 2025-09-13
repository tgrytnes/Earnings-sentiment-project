from IPython.display import HTML


def load_custom_styles():
    return HTML(
        """
    <style>
    div.custom-header {
      padding: 18px;
      border: 2px solid #e8eaed;
      border-radius: 12px;
      background: #fcfcff;
    }
    div.custom-header h1 {
      margin: 0;
      color: #1a73e8;
    }
    div.custom-header p {
      margin: 6px 0 0 0;
      color: #5f6368;
    }
    </style>
    """
    )

