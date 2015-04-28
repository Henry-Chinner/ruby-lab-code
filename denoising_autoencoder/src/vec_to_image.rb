require 'chunky_png'
require './data_loader.rb'

class VecToImage

	def self.viz_hidden_layer(wide,high,matrix)
	
	@images_wide = wide
	@images_high = high
	
	image = ChunkyPNG::Image.new(@images_wide * 28,@images_high * 28)
	
	@images_wide.times do |i|
		puts i
		@images_high.times do |j|
			puts j

		hidden = matrix[i * @images_wide + j]
		
		den = (hidden.inject(0){|sum,el| sum = sum + (el ** 2)}) ** (0.5)

		hidden.map!{|el| el / den}

		min = hidden.min
		max = hidden.max

		hidden.map!{|el| (((el - min) / (max-min)) * 255).ceil}

			hidden.size.times do |k|

				#colored version -- image[(i)*(28) + (k % 28),(j)*(28) + (k / 28)] = ChunkyPNG::Color.rgb((small_image[k] * s2).ceil,(small_image[k] * s3).ceil,(small_image[k] * s1).ceil)
				image[(i)*(28) + (k % 28),(j)*(28) + (k / 28)] = ChunkyPNG::Color.grayscale(hidden[k])
			
			end
		end
	end

	image.save('mnist.png')

	end



end



