require 'pry'

class CircularArray < Array
  def [](*params)
    get_set(:[], *params)
  end

  def []=(*params)
    get_set(:[]=, *params)
  end

private
  def get_set(method, *params)
    i = params.shift
    if i.is_a? Integer
      Array.instance_method(method).bind(self).call(circular_int(i), *params)
    elsif i.is_a? Range
      c = circular_range(i)
      if c.is_a? Range
        Array.instance_method(method).bind(self).call(c, *params)
      else
        Array.instance_method(method).bind(self).call(c[0], *params) + Array.instance_method(method).bind(self).call(c[1], *params)
      end
    end
  end

  def circular_int(i)
    (i + self.length) % self.length
  end

  def circular_range(r)
    b = circular_int(r.begin)
    e = circular_int(r.end)
    if b > e
      [b..length-1, 0..e]
    else
      b..e
    end
  end
end

class Board
  EMPTY_CELL = ' . '
  attr_accessor :board, :position, :food_gradient
  def initialize(height: 40, width: 40)
    @height = height
    @width = width
    @board = CircularArray.new(height){ CircularArray.new(width, EMPTY_CELL) }
    @position = {}

    20.times{ spawn(EnergySystem) }
    20.times{ spawn(FoodSystem) }
    20.times{ spawn(FoodParticle) }

    refresh_food_gradient
  end

  def draw
    puts "\e[H\e[2J"
    draw_board(@board)
    # puts '---'
    # draw_food_gradient(@food_gradient)
  end

  def step
    refresh_food_gradient
    items_with_step = []
    @board.each do |row|
      row.each do |item|
        items_with_step << item if item.respond_to?(:step)
      end
    end

    items_with_step.each {|item| item.step(self) }

    draw
  end

  def move_item(item, row_index, column_index)
    if board[row_index][column_index].is_a? String
      # clear current position
      current_row_index, current_column_index = @position[item]
      raise "Item not on board" if current_row_index.nil? || current_column_index.nil?
      board[current_row_index][current_column_index] = EMPTY_CELL

      # move to new position
      board[row_index][column_index] = item
      @position[item] = [row_index, column_index]
    else
      # raise "Can't move to a place that has an item"
    end
  end

  def delete_item(item)
    current_row_index, current_column_index = @position[item]
    raise "Item not on board" if current_row_index.nil? || current_column_index.nil?

    @board[current_row_index][current_column_index] = EMPTY_CELL
    @position[item] = nil
  end

private
  def refresh_food_gradient
    food_gradient_radius = 5

    @food_gradient = CircularArray.new(@height){ CircularArray.new(@width, 0) }

    @board.each_with_index do |row, row_index|
      row.each_with_index do |item, column_index|
        if item.is_a? FoodParticle
          (row_index-food_gradient_radius..row_index+food_gradient_radius).each do |fgr|
            (column_index-food_gradient_radius..column_index+food_gradient_radius).each do |fgc|
              g = food_gradient_radius - ((row_index-fgr).abs + (column_index - fgc).abs)
              @food_gradient[fgr][fgc] += g if g > 0
            end
          end
        end
      end
    end
  end

  def draw_board(b)
    b.each do |row|
      puts row.join('')
    end
  end

  def draw_food_gradient(f)
    f.each do |row|
      puts row.collect{|i| "%3d" % i }.join('')
    end
  end

  def spawn(klass, h = rand(@height), w = rand(@width))
    instance = klass.new
    @board[h][w] = instance
    @position[instance] = [h, w]
  end
end

class System
  def initialize
    @energy = 10
  end

  def check_alive(board)
    if @energy <= 0
      board.delete_item(self)
    end
  end
end

class EnergySystem < System
  def to_s
    '[E]'
  end
end

class FoodSystem < System
  def to_s
    '[F]'
  end

  def step(board)
    row_index, column_index = board.position[self]
    return if row_index.nil? || column_index.nil?

    potential_next_positions = {
      # [row_index - 1, column_index - 1] => board.food_gradient[row_index - 1][column_index - 1],
      [row_index - 1, column_index    ] => board.food_gradient[row_index - 1][column_index    ],
      # [row_index - 1, column_index + 1] => board.food_gradient[row_index - 1][column_index + 1],

      [row_index    , column_index - 1] => board.food_gradient[row_index    ][column_index - 1],
      [row_index    , column_index + 1] => board.food_gradient[row_index    ][column_index + 1],

      # [row_index + 1, column_index - 1] => board.food_gradient[row_index + 1][column_index - 1],
      [row_index + 1, column_index    ] => board.food_gradient[row_index + 1][column_index    ],
      # [row_index + 1, column_index + 1] => board.food_gradient[row_index + 1][column_index + 1],
    }

    max_food_gradient = potential_next_positions.values.max

    next_positions = potential_next_positions.select{|k, v| v == max_food_gradient}
    next_position = next_positions.keys.sample
    # puts "self: #{to_s}"
    # puts "Current #{row_index}, #{column_index} ==> next #{next_position.join(', ')}"
    board.move_item(self, *next_position)
    @energy -= 1

    # check_alive(board)
  end
end

class Particle
end

class FoodParticle < Particle
  def to_s
    ' f '
  end
end

srand 1234
b = Board.new

b.draw
50.times do
  b.step
  sleep 0.05
end

# (0..20).each do |s|
#   puts "\e[H\e[2J"
#   b.board[s..s-1].each do |row|
#     puts row.join('')
#   end
#   sleep 0.01
# end
