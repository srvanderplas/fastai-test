
library(XML)
library(xml2)
library(tidyverse)

document <- read_xml("Original Data/Annotations/yellow-box-wendal-blush_product_9207902_color_5249.xml")
list1 <- as_list(document)
res <- xmlParse("Original Data/Annotations/1-state-hedde-2-whiskey-multi-leopard-haircalf_product_9144894_color_784454.xml")
list2 <- xmlToList(res)


node <- xml_find_all(document, "object")[[1]]
poly <- xml_find_all(node, "polygon")


# using read_xml and some node parsing...
get_polygon <- function(poly) {
  tibble(
    user = xml_find_all(poly, "username")  %>% map_chr(xml_text),
    points = xml_find_all(poly, "pt") %>% map_df(., ~{
      tmp <- tibble(x = xml_find_all(., "x") %>% xml_text() %>% as.numeric(),
                    y = xml_find_all(., "y") %>% xml_text() %>% as.numeric())
    }) %>% list()
  )
}

get_objects <- function(node) {
  polygons = get_polygon(xml_find_all(node, "polygon"))
  tibble(
    labels = xml_find_first(node, "name") %>% xml_text() %>% str_split(","),
    date = xml_find_first(node, "date") %>% xml_text(),
    id = xml_find_first(node, "id") %>% xml_text(),
    type = xml_find_first(node, "type") %>% xml_text(),
    user = polygons$user,
    polygon = polygons$points
  )
}

xml_find_all(document, "object") %>% map_df(get_objects)
