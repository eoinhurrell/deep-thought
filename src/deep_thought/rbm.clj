(ns deep-thought.rbm
  (:use clojure.core.matrix)
  (:require
   [incanter.core]
   [incanter.distributions :as D]
   [clojure.core.matrix.operators :as M]))

(defn compute-sigmoid
  "Get sigmoid of a matrix"
  [x]
  (M// 1 (M/+ 1 (exp (M/- x)))))

(defn sample-layer
  [])

(defn cycle-fn
  "Function to take one step in RBM learning"
  [rbm learning-rate]
  ;; hidden 0 input (matrix mul prev-res wts) propagate down
  ;; hidden 0 mean (compute sigmoid) compute hidden's activation
  ;; hidden 0 (sample bernoulli) sample visible given hidden's activation
  ;; hidden 0 out (output?)
  ;; visible 0 input (:down (matrix mul prev-res (transpose wts)) propagate up
  ;; visible 0 mean (compute sigmoid) compute visible's activation
  ;; visible 0 out (output?)
  ;; hidden 1 input (matrix mul prev-res wts) propagate down
  ;; hidden 1 mean (compute sigmoid) recompute hidden's activation
  ;; step output (output?)
  )

;; This should just be running the rbm matrix through the
;; step-fn epoch times
(defn fit
  "Train RBM on available data"
  [training epoch])

(defn predict
  "Use trained RBM to predict new data"
  [dataset])
