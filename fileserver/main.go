package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/glassdoor_reviews.csv", func(w http.ResponseWriter, r *http.Request) {
		fmt.Println("Serving browser download")
		http.ServeFile(w, r, "./glassdoor_reviews.csv")
	})

	http.ListenAndServe(":8101", nil)
}