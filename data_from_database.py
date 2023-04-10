import pymysql.cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             database='autotag-crawl',
                             autocommit=True,
                             )

def get_document_and_word():
  # result = ""
  with connection:
      # Create table
      # with connection.cursor() as cursor:
      #     sql = "CREATE TABLE `users` (`id` int(11) NOT NULL AUTO_INCREMENT, `email` varchar(255) COLLATE utf8_bin NOT NULL, `password` varchar(255) COLLATE utf8_bin NOT NULL, PRIMARY KEY (`id`))"
      #     cursor.execute(sql)

      # with connection.cursor() as cursor:
      #     # Create a new record
      #     sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
      #     cursor.execute(sql, ('webmaster@python.org', 'very-secret'))

      # connection is not autocommit by default. So you must commit to save
      # your changes.
      # connection.commit()

      with connection.cursor() as cursor:
          # Read a single record
          
          cursor.execute("SELECT page_information.title, page_information.content_article FROM `page_information` WHERE page_information.content_article IS NOT NULL LIMIT %s", (14))
          resulta = cursor.fetchall()
          
          cursor.execute("SELECT page_tags.tag, page_information.title FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page LIMIT %s, %s", (1, 100))
          resultb = cursor.fetchall()
          
          return resulta, resultb

          
# def get_document_and_tag():
#   with connection:
#       # Create table
#       # connection is not autocommit by default. So you must commit to save
#       # your changes.
#       # connection.commit()

#       with connection.cursor() as cursor:
#           # Read a single record
#           cursor.execute("SELECT page_tags.tag,page_information.title FROM `page_tags` INNER JOIN page_information ON page_tags.page_id = page_information.id_page LIMIT %s, %s", (1, 100))
#           result = cursor.fetchall()
#           return result
#           # print(result)
  
#   # connection.close()