from icrawler.builtin import GoogleImageCrawler

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # 경고 메세지 처리

attractionList = ['tower eiffel', 'statue of liberty', 'niagara falls', 'colosseum', 'pyramid']
attractionFolderList = ['eiffel', 'liberty', 'niagara', 'colosseum', 'pyramid']

for idx, val in enumerate(attractionList):
    google_crawler = GoogleImageCrawler(
        feeder_threads=10,
        parser_threads=10,
        downloader_threads=10,
        storage={'root_dir': 'data_orign/' + attractionFolderList[idx]})
    google_crawler.session.verify = False
    filters = dict(type='photo')
    google_crawler.crawl(keyword=val, filters=filters, max_num=1000, file_idx_offset=0)