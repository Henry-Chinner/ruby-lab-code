require 'pry'

module NNFunctions

  def scalar(mat,scalar)
    mat * scalar
  end

  def scalar_shift(mat,scalar,shift)
    (mat * scalar) - (NMatrix.ones(mat.shape) * shift)
  end  

  def add_noise(mat,noise_level)

    mat.each_row do |row|
      row.map!{|el| rand > noise_level ? el : 0}
    end
  
    mat 
  end

  def shift_mat(mat)
    bias_col = NMatrix.new([mat.shape[0],1], 1, {:dtype => :float32, :default => 1})
    bias_mat = NMatrix.new([mat.shape[0],mat.shape[1]+1], 1, {:dtype => :float32})
    bias_mat[0..(mat.shape[0]-1),0..(mat.shape[1])-1] = mat
    bias_mat[0..(mat.shape[0]-1),mat.shape[1]] = bias_col
    bias_mat
  end 

  def sigmoid(mat)
    NMatrix.ones(mat.shape) / (NMatrix.ones(mat.shape) + (-mat).exp)
  end 

  def tanh(mat)
    mat.tanh
  end

  def relu(mat)
    mat.map! do |el|
      if el < 0
        0.0
      else
        el
      end
    end     
    mat       
  end  

  def sine(mat)
    mat.sin
  end

  def softmax(mat)
    mat.map!{|el| Math::exp(el) }
    sum = mat.sum(1)
    i = 0
    ret_mat = mat.each_row do |row|
      row.map{|el| el / sum[i].to_f}
      i +=1    
    end

    ret_mat
  end 

  def activation_function(mat,func)
    if func[:name] == 'sin'
      return  mat.sin
    elsif func[:name] == 'sigmoid'
      return sigmoid(mat)
    elsif func[:name] == 'tanh'
      return tanh(mat)
    elsif func[:name] == 'softmax'
      return softmax(mat)
    elsif func[:name] == 'scalar'
      return scalar(mat,func[:params][:scalar])
    elsif func[:name] == 'scalar_shift'
      return scalar_shift(mat,func[:params][:scalar],func[:params][:shift])
    elsif func[:name] == 'relu'
      return relu(mat)
    elsif func[:name] == 'linear'
      return mat                
    end
  end

  def derivative(mat,func,function_defined = false)
    if func[:name] == 'sin'
      return mat.cos

    elsif func[:name] == 'sigmoid'
      if function_defined == false
        return d_sig_z(mat)
      else
        return d_sig_a(mat)
      end  

    elsif func[:name] == 'tanh'
      if function_defined == false
        return d_tanh_z(mat)
      else
        return d_tanh_a(mat)  
      end

    elsif func[:name] == 'relu'
      return d_relu(mat)  

    elsif func[:name] == 'softmax'
      return d_softmax(mat)

    elsif func[:name] == 'linear'
      return NMatrix.ones(mat.shape)    

    end      
  end

  def d_tanh_z(mat)
     NMatrix.ones(mat.shape)- (tanh(mat) ** 2)  
  end

  def d_tanh_a(mat)
     NMatrix.ones(mat.shape)- (mat ** 2)  
  end

  def d_sig_z(mat)
    (sigmoid(mat) * (NMatrix.ones(mat.shape) - sigmoid(mat)))
  end

  def d_sig_z(mat)
     (mat * (NMatrix.ones(mat.shape) - mat))
  end

  def d_softmax(mat)
    mat.map!{|el| Math::exp(el) }
    sum = mat.sum(1)
    i = 0
    ret_mat = mat.each_row do |row|
      row.map{|el| ( (el * sum[i] - (el ** 2)) /(2*sum[i]) )}
      i +=1    
    end
   
    ret_mat
  end  

  def d_relu(mat)
    mat.map! do |el|
      if el < 0
        0.0
      else
        1.0
      end
    end      
    mat  
  end  


end