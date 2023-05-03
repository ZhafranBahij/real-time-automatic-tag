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
          
          # Mengambil data untuk title dan content article
          cursor.execute("SELECT DISTINCT page_information.title, page_information.content_article FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page")
          resulta = cursor.fetchall()
          
          # Mengambil data untuk tag dan title
          cursor.execute("SELECT DISTINCT page_tags.tag, page_information.title FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page")
          resultb = cursor.fetchall()
          
          return resulta, resultb

        