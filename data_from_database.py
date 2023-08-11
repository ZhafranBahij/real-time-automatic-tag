import pymysql.cursors

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             database='autotag-crawl',
                             autocommit=True,
                             )

# Mengambil data dari database
def get_data():
  with connection:

      with connection.cursor() as cursor:
          
          # Mengambil data tag, title, dan isi artikel
          cursor.execute("SELECT DISTINCT page_tags.tag, page_information.id_page, page_information.content_article, page_information.title FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page")
          result = cursor.fetchall()
          return result

        