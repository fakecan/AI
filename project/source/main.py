def main():
    import application
    application.run()

# def clear_log():
#     import logging
#     import os
#     import tensorflow as tf

#     logger = logging.getLogger('chardet')
#     logger.setLevel(logging.CRITICAL)
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ = '__main__':
    print('AI loading...')
    print('여행지 안내 챗봇(지역 → 관광지 → 교통수단 → 날씨)')
    main()