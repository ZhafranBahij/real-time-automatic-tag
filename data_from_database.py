import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             database='autotag-crawl',
                             autocommit=True,
                             )

def get_data():
  # result = ""
  with connection:

      with connection.cursor() as cursor:
          
          # Mengambil data tag, title, dan isi artikel
          cursor.execute("SELECT DISTINCT page_tags.tag, page_information.id_page, page_information.content_article, page_information.title FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page")
          # cursor.execute("SELECT DISTINCT page_tags.tag, page_information.title FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page LIMIT 21")
          # cursor.execute("SELECT DISTINCT page_tags.tag FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page LIMIT 21")

          result = cursor.fetchall()
          
          return result

        