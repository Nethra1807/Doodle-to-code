def map_class_to_html(predicted_class):
    """
    Maps the predicted class name to its corresponding HTML code snippet.
    """
    mappings = {
        'Button': '<button style="padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer;">Click Me</button>',
        
        'Checkbox': '<label><input type="checkbox"> Checkbox Label</label>',
        
        'Radio': '<label><input type="radio" name="radio-group"> Radio Option</label>',
        
        'Table': """
<table border="1" style="width: 100%; border-collapse: collapse;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>Header 1</th>
      <th>Header 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data 1</td>
      <td>Data 2</td>
    </tr>
  </tbody>
</table>
        """.strip(),
        
        'Form': """
<form style="display: flex; flex-direction: column; gap: 10px; max-width: 300px;">
  <input type="text" placeholder="Enter Name" style="padding: 8px;">
  <input type="email" placeholder="Enter Email" style="padding: 8px;">
  <button type="submit" style="padding: 10px; background-color: #28a745; color: white; border: none;">Submit</button>
</form>
        """.strip(),
        
        'alert': """
<div style="padding: 15px; background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; border-radius: 4px;">
  <strong>Warning!</strong> This is an alert message.
</div>
        """.strip(),
        
        'switch_disabled': """
<div style="display: flex; align-items: center; gap: 8px; opacity: 0.5;">
  <div style="width: 40px; height: 20px; background-color: #ccc; border-radius: 20px; position: relative;">
    <div style="width: 16px; height: 16px; background-color: white; border-radius: 50%; position: absolute; top: 2px; left: 2px;"></div>
  </div>
  <span style="color: #666; cursor: not-allowed;">Disabled Switch</span>
</div>
        """.strip(),
        
        'switch_enabled': """
<div style="display: flex; align-items: center; gap: 8px;">
  <div style="width: 40px; height: 20px; background-color: #4CAF50; border-radius: 20px; position: relative;">
    <div style="width: 16px; height: 16px; background-color: white; border-radius: 50%; position: absolute; top: 2px; right: 2px;"></div>
  </div>
  <span style="color: #333; cursor: pointer;">Enabled Switch</span>
</div>
        """.strip(),
        
        'radio_button_unchecked': """
<label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
  <input type="radio" style="width: 18px; height: 18px;" disabled> Unchecked Radio
</label>
        """.strip(),
        
        'radio_button_checked': """
<label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
  <input type="radio" style="width: 18px; height: 18px;" checked> Checked Radio
</label>
        """.strip()
    }
    
    return mappings.get(predicted_class, f"<!-- Component Not Supported: {predicted_class} -->")
