(ns deep-thought.mlp
  (:use clojure.core.matrix)
  (:use clojure.test))
;; Heavily influenced by implementation http://www.fatvat.co.uk/

(def hidden-layers 50)
(def learning-rate 0.5)
(def momentum 0.1)

(def activation-fn
  (fn [x] (Math/tanh x)))
(def activation-fn-derivation
  (fn [y] (- 1.0 (* y y))))

(defstruct bp-nn
  :weight-input
  :weight-output
  :change-input
  :change-output)

(defn zero-matrix
  "Create a matrix (list of lists)"
  [height width]
  (repeat width (repeat height 0)))

(defn emap
  "Apply a fn to every element in a matrix"
  [m func]
  (map
   (fn [x] (map func x))
   m))

(defn rand-range 
  "Return a real number within the given range"
  [l h]
  (+ (rand (- h l)) l))

(defn create-network
  "Create a network with the given number of input, hidden and output nodes"
  ([input hidden output]
   (create-network input hidden output true))
  ([input hidden output use-random-weights]
   (let [i (inc input)
         w-func (if use-random-weights
                  (fn [_] (rand-range -0.2 0.2))
                  (fn [_] 0.2))
         o-func (if use-random-weights
                  (fn [_] (rand-range -2.0 2.0))
                  (fn [_] 2.0))]
     (struct bp-nn
             (emap (zero-matrix hidden i) w-func)
             (emap (zero-matrix output hidden) o-func)
             (zero-matrix hidden i)
             (zero-matrix output hidden)))))

(defn calculate-hidden-deltas
  "Calculate the error terms for the hidden"
  [wo ah od]
  (let [errors
        (map
         (partial reduce +)
         (map (fn [x] (map * x od)) wo))]
    (map (fn [h e] (* e (activation-fn-derivation h))) ah errors)))
    
(defn update-weights
  [w deltas co ah]
  (let [x
        (map
         (fn
           [wcol ccol h]
           (map
            (fn
              [wrow crow od]
              (let [change (* od h)]
                [(+ wrow
                    (* learning-rate change)
                    (* momentum crow))
                 change]))
            wcol ccol deltas))
         w co ah)]
    (println x)
    [(emap x first) (emap x second)]))

(defn apply-activation-fn
  "Calculate the hidden activations"
  [w i]
  (apply map (comp activation-fn +) (map (fn [col p] (map (partial * p) col)) w i)))

(defn run-network
  "Run the network with the given pattern and return the output and the hidden values"
  [pattern network]
  (assert (= (count pattern) (dec (count (network :weight-input)))))
  (let [p (cons 1 pattern)] ;; ensure bias term added
    (let [wi (network :weight-input)
	  wo (network :weight-output)
	  ah (apply-activation-fn wi p)
	  ao (apply-activation-fn wo ah)]
      [ao ah])))

(defn back-propagate
  "Back propagate the results to adjust the rates"
  [target p results network]
  (assert (= (count target) (count (first (get network :weight-output)))))
  (let [pattern (cons 1 p) ;; ensure bias term added
        ao (first results)
        ah (second results)
        error (map - target ao)
        wi (network :weight-input)
        wo (network :weight-output)
        ci (network :change-input)
        co (network :change-output)
        output-deltas (map (fn [o e] (* e (activation-fn-derivation o))) ao error)
        hidden-deltas (calculate-hidden-deltas wo ah output-deltas)
        updated-output-weights (update-weights wo output-deltas co ah)
        updated-input-weights (update-weights wi hidden-deltas ci pattern)]
    (struct bp-nn
            (first  updated-input-weights)
            (first  updated-output-weights)
            (second updated-input-weights)
            (second updated-output-weights))))

(defn run-patterns
  [network samples expecteds]
  (reduce 
   (fn [n expectations] 
     (let [[sample expected] expectations
	   [ah ao] (run-network sample n)]
       (back-propagate expected sample [ah ao] n)))   
   network ;; initial value
   (map list samples expecteds)))  

(defn train-network
  ([samples expected]
     (train-network (create-network (count (first samples)) 
				    hidden-layers (count (first expected))) 
		    samples expected))
  ([network samples expected]
     (iterate (fn [n] (run-patterns n samples expected)) network)))
