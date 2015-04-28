require './layer.rb'
require 'fileutils'
require 'csv'


class FFNetwork
  attr_reader :network, :error_history

  def initialize(ary)
    @network = ary
    @y_index = @network.find_index {|layer| layer.name == @network[-1].y_pointer[:l_name]}
    @error_history = []
    @classification_history = []
  end  

  def forward

    0.upto(@network.size-1) do |i|
      if i == 0  
        @network[i].pull_z
        @network[i].calc_activation
        @network[i].add_bias  
      else
        @network[i].z = @network[i-1].a_with_bias.dot(@network[i].w)
        @network[i].a = @network[i].calc_activation
          if i != (@network.size-1)
            @network[i].add_bias
          end
      end
    end    
  end
  
  def backprop(alpha)

    (@network.size-1).downto(1) do |i|
      
      if i == (@network.size - 1)

        @network[i].y = @network[@y_index].send @network[-1].y_pointer[:l_attr]
        @network[i].error # calc error
        @network[i].delta = @network[i-1].a_with_bias.transpose.dot(@network[i].grad) /  @network[0].batch_size.to_f
      elsif i != (@network.size - 1)
        @network[i].grad = @network[i+1].grad.dot(@network[i+1].w.transpose)[0..(@network[0].batch_size-1),0..(@network[i].fan_out-1)] * @network[i].calc_derivative
        @network[i].delta = @network[i-1].a_with_bias.transpose.dot(@network[i].grad) /  @network[0].batch_size.to_f
      end    
    end

    @network.each do |layer| 
      layer.update_weights(alpha)
    end  

  end

  def train(opts ={})
    opts[:max_epocs].times do |i|
     
      forward
      backprop(opts[:alpha])
      
      if i % 100 == 0 and @network.size == 3
        @network[1].viz_weights
      end

      @error_history << ( (@network[-1].y - network[-1].a).abs.sum.inject(0){|sum,el| sum +=el} / @network[0].batch_size.to_f )
      
      temp_ary = []
      j = 0

      if @network[-1].name == 'classifier'
        @network[-1].a.each_row do |row|
          temp_ary << (row.each_with_index.max[1] == @network[-1].y.row(j).each_with_index.max[1] ? 1.0 : 0.0)
          j += 1
        end
        @classification_history << temp_ary.inject(0){|sum,el| sum+=el} / temp_ary.size.to_f 
      end

      
    
      ave_error_history = running_average(1000,@error_history)
      ave_error_history_5000 = running_average(5000,@error_history)

      if @network[-1].name == 'classifier'
        ave_classification_history = running_average(1000,@classification_history)
        ave_classification_history_5000 = running_average(5000,@classification_history)
        ratio = (ave_classification_history  / ave_classification_history_5000)
      end

      puts "Running Average Error (1000) => #{ave_error_history}"
      puts "Running Average Error (5000) => #{ave_error_history_5000}"

      if @network[-1].name == 'classifier'
        puts "Running Average Classification (1000) => #{ave_classification_history} "
        puts "Running Average Classification (5000) => #{ave_classification_history_5000}"
        puts "Classification Runninge Average Ratio => #{ratio}"
      end

      puts "Batch number => #{i}"
      puts "------------------------------------"

      # break if (ratio < 1) and (i > opts[:max_epocs])
        
    end
  end

  def prep_kaggle_mnist
    @network[0].dt = DataTable.new({:file => '../data/test.csv' , :label_index => :none})
    @network[0].set_data_cursor
    CSV.open("../data/submission.csv", "wb") do |csv|
      csv << ["ImageID", "Label"]
      @network[0].dt.observations.size.times do |i|
        puts i
        forward
        csv << [(i+1).to_s,  @network[-1].get_classification ]
      end
    end
   
    @network[0].load_data #load back training data
  end  

  def running_average(scale,ary)
    ary.last(scale).inject{ |sum, el| sum + el}.to_f / [scale,ary.size].min 
  end 

  def self.load_model(model_name)
      Marshal.load(File.binread("../data/models/#{model_name}"))
  end

  def wake_up_layers
    @network.each do |layer|
      layer.wake_up
    end  
  end  

  def persist(model_name)
    dirname = File.dirname("../data/models/")
    unless File.directory?("../data/models/")
      FileUtils.mkdir_p("../data/models/")
    end

      @network.each do |layer|
        layer.prep_persist
      end
      File.open("../data/models/#{model_name}.txt",'w+'){|f| f << Marshal.dump(self)}
  end

end