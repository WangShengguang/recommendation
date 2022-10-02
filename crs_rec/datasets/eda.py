import os

import pandas as pd
from pandas_profiling import ProfileReport

cur_dir = os.path.dirname(os.path.abspath(__file__))


def get_data_df():
    file_path = os.path.join(cur_dir, 'chat_sessions_20220320.txt')
    data_df = pd.read_csv(file_path,
                          names=['appid', 'session_id', 'user_id', 'datetime',
                                 'user_utterance', 'bot_utterance', 'NULL', 'skill'],
                          sep='\t')
    data_df.dropna(how='all', inplace=True)
    return data_df


def eda():
    data_df = get_data_df()
    print(data_df.shape)
    data_df.fillna('', inplace=True)
    data_df['user_utterance_len'] = data_df['user_utterance'].apply(len)
    data_df['bot_utterance_len'] = data_df['bot_utterance'].apply(len)
    #
    profile = ProfileReport(data_df, title="data_df_Report")
    profile.to_file(f"data_df.html")  #
    #
    # session_df = pd.DataFrame(data_df.groupby('session_id').count().reset_index(name="count"))
    # session_df = data_df.groupby('session_id').count().reset_index(inplace=True)
    session_df = data_df.groupby('session_id').count()['skill'].reset_index(name='session_len')
    profile = ProfileReport(session_df, title="session_df_Report")
    profile.to_file(f"session_df.html")  #
    # breakpoint()


def analysis():
    data_df = get_data_df()


def main():
    eda()


if __name__ == '__main__':
    main()
